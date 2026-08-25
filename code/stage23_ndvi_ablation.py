"""
Stage 23 (P0-4 follow-up): NDVI temporal ablation - does NDVI source year matter?
Compare 11-factor models with: (a) 2020 NDVI (current), (b) 2007 pre-quake NDVI.
Report: random-CV AUC/ACC/F1/Kappa for both, NDVI importance change, and
XGBoost spatial-CV (Scheme B) for both.
Addresses the meta-review point that 2020 NDVI postdates the 2008 landslide labels.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
SEED = 42
FACTORS = ["elev", "slope", "aspect_sin", "aspect_cos", "plan_c", "prof_c",
           "twi", "spi", "sti", "relief", "ndvi"]
FEATURES = FACTORS + ["dist_stream"]

def load_samples(ndvi_path):
    samples = pd.read_csv(os.path.join(OUT, "samples_raw2.csv"))
    with rasterio.open(os.path.join(OUT, "elev.tif")) as s0:
        tf = s0.transform
    with rasterio.open(os.path.join(OUT, "dist_stream_low.tif")) as s1:
        tf_low = s1.transform
    rasters = {f: rasterio.open(os.path.join(OUT, f + ".tif")) for f in FACTORS}
    rasters["ndvi"] = rasterio.open(ndvi_path)
    rasters["dist_stream"] = rasterio.open(os.path.join(OUT, "dist_stream_low.tif"))
    lon = samples["lon"].values; lat = samples["lat"].values
    def extract(src, tr, lo, la):
        arr = src.read(1)
        c = (lo - tr.c) / tr.a; r_ = (tr.f - la) / (-tr.e)
        vals = []
        for ci, ri in zip(c, r_):
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
    return df, df[FEATURES].values, df["label"].values

def run_random_cv(X, y):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    models = {
        "LR": LogisticRegression(max_iter=3000, random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                     n_jobs=2, random_state=SEED),
        "XGB": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=SEED, n_jobs=2),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {}
    for name, mdl in models.items():
        aucs, accs, f1s, kaps = [], [], [], []
        for tr, te in cv.split(X, y):
            Xtr, Xte = (Xs[tr], Xs[te]) if name == "LR" else (X[tr], X[te])
            mdl.fit(Xtr, y[tr])
            yp = mdl.predict(Xte); pr = mdl.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[te], pr)); accs.append(accuracy_score(y[te], yp))
            f1s.append(f1_score(y[te], yp)); kaps.append(cohen_kappa_score(y[te], yp))
        out[name] = {"AUC": f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}",
                     "Acc": f"{np.mean(accs):.3f}±{np.std(accs):.3f}",
                     "F1": f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}",
                     "Kappa": f"{np.mean(kaps):.3f}±{np.std(kaps):.3f}"}
    return out

def ndvi_importance(X, y, feat_names):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                          random_state=SEED, n_jobs=2).fit(X, y)
    imp = m.feature_importances_
    return dict(zip(feat_names, imp))

print("=== A) NDVI 2020 (current) ===")
df20, X20, y20 = load_samples(os.path.join(OUT, "ndvi.tif"))
r20 = run_random_cv(X20, y20)
for k, v in r20.items():
    print(k, v)
i20 = ndvi_importance(X20, y20, FEATURES)
print("ndvi importance (2020):", round(i20["ndvi"], 4))
print("ndvi rank (2020):", sorted(FEATURES, key=lambda f: -i20[f]).index("ndvi") + 1)

print("\n=== B) NDVI 2007 (pre-quake) ===")
ndvi07_path = os.path.join(OUT, "ndvi_2007.tif")
assert os.path.exists(ndvi07_path), "ndvi_2007.tif missing - run stage20 first"
df07, X07, y07 = load_samples(ndvi07_path)
r07 = run_random_cv(X07, y07)
for k, v in r07.items():
    print(k, v)
i07 = ndvi_importance(X07, y07, FEATURES)
print("ndvi importance (2007):", round(i07["ndvi"], 4))
print("ndvi rank (2007):", sorted(FEATURES, key=lambda f: -i07[f]).index("ndvi") + 1)

summary = {
    "ndvi_2020": {"model_results": r20, "ndvi_importance": round(i20["ndvi"], 4),
                  "ndvi_rank": sorted(FEATURES, key=lambda f: -i20[f]).index("ndvi") + 1},
    "ndvi_2007": {"model_results": r07, "ndvi_importance": round(i07["ndvi"], 4),
                  "ndvi_rank": sorted(FEATURES, key=lambda f: -i07[f]).index("ndvi") + 1},
}
import json
with open(os.path.join(OUT, "ndvi_ablation.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nsaved ndvi_ablation.json")