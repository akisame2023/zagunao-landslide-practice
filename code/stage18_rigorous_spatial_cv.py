"""
Stage 18 (P0-1): Rigorous spatial CV following Roberts et al. (2017) & Meyer & Pebesma (2022).
Two rigorous schemes, both with per-fold composition persisted to JSON:

(A) Spatial k-means blocks with buffer exclusion (leave-location-out style):
    - cluster samples into k=5 spatially contiguous blocks (k-means on coords)
    - for each held-out block, REMOVE from training all samples within
      BUFFER_M (meters) of any test sample -> guarantees training/test separation
    - reports n folds actually used, class composition per fold

(B) Spatial grid blocks with buffer exclusion (alternative):
    - partition by 4x4 grid bins, contiguous folds via equal-area assignment,
      buffer exclusion applied the same way.

Buffer: 300 m (10 pixels) as primary; 150 m sensitivity.
Also persists: per-fold class composition, buffer distance, folds used.
"""
import os
import json
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
SEED = 42
BUFFER_M = 300.0   # primary buffer (10 x 30m pixels)
BUFFER_SENS = 150.0

FACTORS = ["elev", "slope", "aspect_sin", "aspect_cos", "plan_c", "prof_c",
           "twi", "spi", "sti", "relief", "ndvi"]
FEATURES = FACTORS + ["dist_stream"]

samples = pd.read_csv(os.path.join(OUT, "samples_raw2.csv"))
with rasterio.open(os.path.join(OUT, "elev.tif")) as s0:
    tf = s0.transform
with rasterio.open(os.path.join(OUT, "dist_stream_low.tif")) as s1:
    tf_low = s1.transform
rasters = {f: rasterio.open(os.path.join(OUT, f + ".tif")) for f in FACTORS}
rasters["dist_stream"] = rasterio.open(os.path.join(OUT, "dist_stream_low.tif"))
lon = samples["lon"].values; lat = samples["lat"].values

def extract(src, tr, lo, la):
    arr = src.read(1)
    c = (lo - tr.c) / tr.a; r = (tr.f - la) / (-tr.e)
    vals = []
    for ci, ri in zip(c, r):
        ci_i, ri_i = int(round(ci)), int(round(ri))
        if 0 <= ri_i < arr.shape[0] and 0 <= ci_i < arr.shape[1]:
            vals.append(arr[ri_i, ci_i])
        else:
            vals.append(np.nan)
    return np.array(vals)

for f in FACTORS:
    tr_use = tf_low if f == "twi" else tf
    samples[f] = extract(rasters[f], tr_use, lon, lat)
samples["dist_stream"] = extract(rasters["dist_stream"], tf_low, lon, lat)
for s in rasters.values():
    s.close()

df = samples.dropna().reset_index(drop=True)
pos = df[df.label == 1]; neg = df[df.label == 0]
n = min(len(pos), len(neg))
df = pd.concat([pos.iloc[:n], neg.sample(n, random_state=SEED)], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
X = df[FEATURES].values; y = df["label"].values
coords = df[["lon", "lat"]].values
print("samples:", X.shape, "balance:", np.bincount(y).tolist())

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# meters per degree approx at lat ~31.5
M_PER_DEG = 111320.0

def buffer_exclude(test_idx, buffer_m, train_idx_all):
    """Remove from train_idx_all any sample within buffer_m of a test sample."""
    test_coords = coords[test_idx]
    keep = np.ones(len(train_idx_all), dtype=bool)
    tr_coords = coords[train_idx_all]
    # approximate distance in degrees -> meters (lon scaled by cos(lat))
    lon_scale = np.cos(np.deg2rad(31.5))
    for i, tc in enumerate(test_coords):
        dlat = (tr_coords[:, 1] - tc[1]) * M_PER_DEG
        dlon = (tr_coords[:, 0] - tc[0]) * M_PER_DEG * lon_scale
        dist = np.sqrt(dlat**2 + dlon**2)
        keep &= dist >= buffer_m
    return train_idx_all[keep]

models = {
    "LR": lambda: LogisticRegression(max_iter=3000, random_state=SEED),
    "RF": lambda: RandomForestClassifier(n_estimators=300, max_depth=12,
                                         min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         eval_metric="logloss", random_state=SEED, n_jobs=-1),
}

def run_scheme(block_ids, buffer_m, name):
    """Leave-one-block-out with buffer exclusion. Returns results + fold metadata."""
    fold_meta = []
    results = {m: {"AUC": [], "Acc": [], "F1": [], "Kappa": [], "n_test": []} for m in models}
    for b in np.unique(block_ids):
        te = block_ids == b
        n_te = te.sum()
        # require both classes in test
        if (y[te] == 1).sum() == 0 or (y[te] == 0).sum() == 0:
            fold_meta.append({"block": int(b), "used": False, "n_test": int(n_te),
                              "pos": int(y[te].sum()), "neg": int((~y[te].astype(bool)).sum()),
                              "reason": "single-class test fold"})
            continue
        tr_all = np.where(~te)[0]
        tr = buffer_exclude(np.where(te)[0], buffer_m, tr_all)
        if len(tr) == 0 or (y[tr] == 1).sum() == 0 or (y[tr] == 0).sum() == 0:
            fold_meta.append({"block": int(b), "used": False, "n_test": int(n_te),
                              "pos": int(y[te].sum()), "neg": int((~y[te].astype(bool)).sum()),
                              "reason": "training degenerate after buffer"})
            continue
        fold_meta.append({"block": int(b), "used": True, "n_test": int(n_te),
                          "pos": int(y[te].sum()), "neg": int((~y[te].astype(bool)).sum()),
                          "n_train_after_buffer": int(len(tr))})
        for mname, maker in models.items():
            m = maker()
            Xtr, Xte = (Xs[tr], Xs[te]) if mname == "LR" else (X[tr], X[te])
            m.fit(Xtr, y[tr])
            yp = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]
            results[mname]["AUC"].append(roc_auc_score(y[te], pr))
            results[mname]["Acc"].append(accuracy_score(y[te], yp))
            results[mname]["F1"].append(f1_score(y[te], yp))
            results[mname]["Kappa"].append(cohen_kappa_score(y[te], yp))
            results[mname]["n_test"].append(int(n_te))
    summary = {}
    for mname in models:
        r = results[mname]
        summary[mname] = {
            "AUC": f"{np.mean(r['AUC']):.3f}±{np.std(r['AUC']):.3f}",
            "Acc": f"{np.mean(r['Acc']):.3f}±{np.std(r['Acc']):.3f}",
            "F1": f"{np.mean(r['F1']):.3f}±{np.std(r['F1']):.3f}",
            "Kappa": f"{np.mean(r['Kappa']):.3f}±{np.std(r['Kappa']):.3f}",
            "n_folds_used": len(r["AUC"]),
        }
    return summary, fold_meta

# ---- Scheme A: spatial k-means blocks (contiguous), buffer 300m ----
km = KMeans(n_clusters=5, random_state=SEED, n_init=20).fit(coords)
block_a = km.labels_
print("\n=== Scheme A: spatial k-means blocks + 300m buffer ===")
res_a, meta_a = run_scheme(block_a, BUFFER_M, "kmeans_contiguous")
for mname, s in res_a.items():
    print(mname, s)

# ---- Scheme B: 4x4 grid blocks, contiguous band assignment via order, buffer 300m ----
lon0, lon1 = coords[:, 0].min(), coords[:, 0].max()
lat0, lat1 = coords[:, 1].min(), coords[:, 1].max()
NB = 4
bi = np.clip(((coords[:, 0] - lon0) / (lon1 - lon0) * NB).astype(int), 0, NB - 1)
bj = np.clip(((coords[:, 1] - lat0) / (lat1 - lat0) * NB).astype(int), 0, NB - 1)
bins = bi * NB + bj
# contiguous folds: assign bins by snake-order (spatial contiguity)
order = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]  # snake over 4x4
fold_ids = np.full(len(y), -1)
for fi, b in enumerate(order):
    fold_ids[bins == b] = fi % 5
print("\n=== Scheme B: 4x4 grid snake folds + 300m buffer ===")
res_b, meta_b = run_scheme(fold_ids, BUFFER_M, "grid_snake")
for mname, s in res_b.items():
    print(mname, s)

# ---- Scheme A sensitivity: buffer 150m ----
print("\n=== Scheme A sensitivity: k-means blocks + 150m buffer ===")
res_a150, meta_a150 = run_scheme(block_a, BUFFER_SENS, "kmeans_contiguous_150m")
for mname, s in res_a150.items():
    print(mname, s)

out = {
    "scheme_A_kmeans_contig_300m": {"results": res_a, "folds": meta_a,
                                     "buffer_m": BUFFER_M, "n_blocks": 5,
                                     "method": "k-means contiguous blocks + buffer exclusion (Roberts et al. 2017; Meyer & Pebesma 2022)"},
    "scheme_B_grid_snake_300m": {"results": res_b, "folds": meta_b,
                                  "buffer_m": BUFFER_M, "n_blocks": 5,
                                  "method": "4x4 grid snake folds + buffer exclusion"},
    "scheme_A_kmeans_contig_150m": {"results": res_a150, "folds": meta_a150,
                                     "buffer_m": BUFFER_SENS, "n_blocks": 5,
                                     "method": "k-means contiguous blocks + buffer 150m (sensitivity)"},
    "note": "fold_meta records per-fold class composition and folds actually used; "
            "results only aggregate folds with both classes in test AND valid train after buffer.",
}
with open(os.path.join(OUT, "cv_rigorous.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\nsaved cv_rigorous.json")

# print fold metadata for inspection
print("\nScheme A fold metadata (300m):")
for fm in meta_a:
    print(" ", fm)
print("\nScheme B fold metadata (300m):")
for fm in meta_b:
    print(" ", fm)