"""
Stage 22 (P0-2): Out-of-fold regional prediction + honest zoning statistics.
Problem being fixed: the original Table 4 "99.56% of landslide samples in very-high
zone" is circular (zoning thresholds applied to in-sample predictions).

Fix: use out-of-fold (OOF) predictions from the rigorous spatial CV (Scheme B,
grid snake folds - the scheme with 4 usable folds), i.e., each sample's probability
comes from a model that never saw its spatial fold. Recompute:
  - OOF AUC per model
  - five-class zoning statistics on OOF probabilities
  - capture rate of 2008 positive samples in OOF-based high/very-high zones
"""
import os
import json
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
SEED = 42
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
nn = min(len(pos), len(neg))
df = pd.concat([pos.iloc[:nn], neg.sample(nn, random_state=SEED)], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
X = df[FEATURES].values; y = df["label"].values
coords = df[["lon", "lat"]].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# ---- reproduce Scheme B folds (grid snake, 300m buffer) ----
lon0, lon1 = coords[:, 0].min(), coords[:, 0].max()
lat0, lat1 = coords[:, 1].min(), coords[:, 1].max()
NB = 4
bi = np.clip(((coords[:, 0] - lon0) / (lon1 - lon0) * NB).astype(int), 0, NB - 1)
bj = np.clip(((coords[:, 1] - lat0) / (lat1 - lat0) * NB).astype(int), 0, NB - 1)
bins = bi * NB + bj
order = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]
fold_ids = np.full(len(y), -1)
for fi, b in enumerate(order):
    fold_ids[bins == b] = fi % 5

M_PER_DEG = 111320.0
def buffer_exclude(test_idx, buffer_m, train_idx_all):
    test_coords = coords[test_idx]
    tr_coords = coords[train_idx_all]
    lon_scale = np.cos(np.deg2rad(31.5))
    keep = np.ones(len(train_idx_all), dtype=bool)
    for tc in test_coords:
        dlat = (tr_coords[:, 1] - tc[1]) * M_PER_DEG
        dlon = (tr_coords[:, 0] - tc[0]) * M_PER_DEG * lon_scale
        dist = np.sqrt(dlat**2 + dlon**2)
        keep &= dist >= 300.0
    return train_idx_all[keep]

models = {
    "LR": lambda: LogisticRegression(max_iter=3000, random_state=SEED),
    "RF": lambda: RandomForestClassifier(n_estimators=300, max_depth=12,
                                         min_samples_leaf=2, n_jobs=2, random_state=SEED),
    "XGB": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="logloss", random_state=SEED, n_jobs=2),
}

prob_oof = {m: np.full(len(y), np.nan) for m in models}
pred_oof = {m: np.full(len(y), np.nan) for m in models}
usage = {m: [] for m in models}

for fold in range(5):
    te = fold_ids == fold
    if te.sum() == 0 or (y[te] == 1).sum() == 0 or (y[te] == 0).sum() == 0:
        print(f"fold {fold}: skipped (single-class test)")
        continue
    tr_all = np.where(~te)[0]
    tr = buffer_exclude(np.where(te)[0], 300.0, tr_all)
    if (y[tr] == 1).sum() == 0 or (y[tr] == 0).sum() == 0:
        print(f"fold {fold}: skipped (degenerate train)")
        continue
    for name, maker in models.items():
        m = maker()
        Xtr, Xte = (Xs[tr], Xs[te]) if name == "LR" else (X[tr], X[te])
        m.fit(Xtr, y[tr])
        prob_oof[name][te] = m.predict_proba(Xte)[:, 1]
        pred_oof[name][te] = m.predict(Xte)
        usage[name].append(fold)
    print(f"fold {fold}: used, n_test={te.sum()}")

print("\n=== OOF performance under spatial Scheme B (grid snake + 300m buffer) ===")
oof_metrics = {}
for m in models:
    valid = ~np.isnan(prob_oof[m])
    auc = roc_auc_score(y[valid], prob_oof[m][valid])
    acc = accuracy_score(y[valid], pred_oof[m][valid])
    f1 = f1_score(y[valid], pred_oof[m][valid])
    kap = cohen_kappa_score(y[valid], pred_oof[m][valid])
    oof_metrics[m] = {"AUC": round(auc, 4), "Acc": round(acc, 4),
                      "F1": round(f1, 4), "Kappa": round(kap, 4),
                      "n": int(valid.sum()), "folds_used": usage[m]}
    print(m, oof_metrics[m])

# ---- honest zoning on OOF probabilities (best model by OOF AUC) ----
best = max(oof_metrics, key=lambda m: oof_metrics[m]["AUC"])
print("\nbest by OOF AUC:", best)
p_oof = prob_oof[best]
valid = ~np.isnan(p_oof)
p = p_oof[valid]; yv = y[valid]
q = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
zones_oof = np.digitize(p, q)
names = ["very_low", "low", "moderate", "high", "very_high"]
rows = []
for k in range(5):
    msk = zones_oof == k
    rows.append({"zone": names[k], "n": int(msk.sum()),
                 "area_share_pct": round(100 * msk.mean(), 2),
                 "pos_share_pct": round(100 * (yv[msk] == 1).mean(), 2) if msk.sum() else np.nan,
                 "n_pos": int((yv[msk] == 1).sum())})
zstat = pd.DataFrame(rows)
zstat.to_csv(os.path.join(OUT, "zoning_oof.csv"), index=False, encoding="utf-8-sig")
print("\n=== Honest OOF zoning (Zagunao samples) ===")
print(zstat.to_string(index=False))

# capture ratio for positives
pos_hi_very = zstat.loc[zstat.zone.isin(["high", "very_high"]), "n_pos"].sum()
print(f"\nPositives in high+very-high (OOF): {pos_hi_very}/{yv.sum()} = "
      f"{100*pos_hi_very/yv.sum():.1f}%")

out = {"oof_metrics": oof_metrics, "zoning_oof": rows, "thresholds": q.tolist()}
with open(os.path.join(OUT, "oof_zoning_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved oof_zoning_summary.json")