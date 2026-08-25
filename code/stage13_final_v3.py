"""
Stage 13 (FINAL v3): 11-factor model = previous 10 + ndvi (dist_fault DROPPED:
GEM global fault dataset coverage at 1:20k-equivalent resolution is too coarse for
this basin; the single mapped fault east of the basin acts as a proxy for
longitude/location rather than physical fault proximity, causing leakage-like
behavior. Documented honestly in the paper).
"""
import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CB = ["#0072B2", "#D55E00", "#009E73"]
OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
SEED = 42

FACTORS = ["elev", "slope", "aspect_sin", "aspect_cos", "plan_c", "prof_c",
           "twi", "spi", "sti", "relief", "ndvi"]
FEATURES = FACTORS + ["dist_stream"]
print("features:", FEATURES)

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
print("final samples:", X.shape, "balance:", np.bincount(y).tolist())

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
models = {
    "LR": LogisticRegression(max_iter=3000, random_state=SEED),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                 n_jobs=-1, random_state=SEED),
    "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=SEED, n_jobs=-1),
}
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
results = []
fig, ax = plt.subplots(figsize=(6.5, 6))
for i, (name, mdl) in enumerate(models.items()):
    aucs, accs, precs, recs, f1s, kappas = [], [], [], [], [], []
    tprs = []
    for tr, te in CV.split(X, y):
        Xtr, Xte = (Xs[tr], Xs[te]) if name == "LR" else (X[tr], X[te])
        mdl.fit(Xtr, y[tr])
        yp = mdl.predict(Xte); proba = mdl.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], proba)); accs.append(accuracy_score(y[te], yp))
        precs.append(precision_score(y[te], yp)); recs.append(recall_score(y[te], yp))
        f1s.append(f1_score(y[te], yp)); kappas.append(cohen_kappa_score(y[te], yp))
        fpr, tpr, _ = roc_curve(y[te], proba); tprs.append((fpr, tpr))
    all_fpr = np.linspace(0, 1, 300)
    mean_tpr = np.mean([np.interp(all_fpr, fpr, tpr) for fpr, tpr in tprs], axis=0)
    mean_tpr[0] = 0
    ax.plot(all_fpr, mean_tpr, lw=2, color=CB[i], label=f"{name} (AUC={np.mean(aucs):.3f})")
    results.append({"model": name,
                    "AUC": f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}",
                    "Accuracy": f"{np.mean(accs):.3f}±{np.std(accs):.3f}",
                    "Precision": f"{np.mean(precs):.3f}±{np.std(precs):.3f}",
                    "Recall": f"{np.mean(recs):.3f}±{np.std(recs):.3f}",
                    "F1": f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}",
                    "Kappa": f"{np.mean(kappas):.3f}±{np.std(kappas):.3f}"})
    print(name, results[-1])
ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (5-fold CV, 11 factors incl. NDVI)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_roc_v3.png"), dpi=300); plt.close(fig)

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT, "model_results_final_v3.csv"), index=False, encoding="utf-8-sig")
print("\n=== FINAL RESULTS (11 factors incl. NDVI) ===")
print(res_df.to_string(index=False))

rf_final = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2,
                                  n_jobs=-1, random_state=SEED).fit(X, y)
xgb_final = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              random_state=SEED, n_jobs=-1).fit(X, y)
imp_df = pd.DataFrame({"feature": FEATURES,
                       "gini": rf_final.feature_importances_,
                       "gain": xgb_final.feature_importances_})
imp_df.to_csv(os.path.join(OUT, "feature_importance_final_v3.csv"), index=False, encoding="utf-8-sig")
print("\n=== IMPORTANCE (11 factors) ===")
print(imp_df.sort_values("gini", ascending=False).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
for ax, col, title in zip(axes, ["gini", "gain"], ["RF (Gini)", "XGBoost (gain)"]):
    d = imp_df.sort_values(col)
    ax.barh(d["feature"], d[col], color="#0072B2")
    ax.set_title(title); ax.set_xlabel("importance")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_importance_v3.png"), dpi=300); plt.close(fig)

best = res_df.loc[res_df["AUC"].str.split("±").str[0].astype(float).idxmax(), "model"]
print("best:", best)
if best == "LR":
    final = LogisticRegression(max_iter=3000, random_state=SEED).fit(Xs, y)
    use_scaler = True
else:
    final = (rf_final if best == "RF" else xgb_final)
    use_scaler = False

rst = {f: rasterio.open(os.path.join(OUT, f + ".tif")) for f in FACTORS}
rst["dist_stream"] = rasterio.open(os.path.join(OUT, "dist_stream_low.tif"))
with rasterio.open(os.path.join(OUT, "elev.tif")) as s0:
    tf2 = s0.transform; ny, nx = s0.shape
with rasterio.open(os.path.join(OUT, "dist_stream_low.tif")) as s1:
    tf_low2 = s1.transform
arrs = {f: rst[f].read(1).astype(np.float64) for f in FACTORS}
arrs["dist_stream"] = rst["dist_stream"].read(1).astype(np.float64)
for s in rst.values():
    s.close()
from rasterio.enums import Resampling
for f in ["twi", "dist_stream"]:
    if arrs[f].shape != (ny, nx):
        with rasterio.open(os.path.join(OUT, ("dist_stream_low" if f == "dist_stream" else f) + ".tif")) as sr:
            arrs[f] = sr.read(1, out_shape=(ny, nx), resampling=Resampling.nearest).astype(np.float64)

rows, cols = np.mgrid[0:ny, 0:nx]
valid = np.ones((ny, nx), dtype=bool)
for f in FEATURES:
    valid &= np.isfinite(arrs[f])
rr, cc = rows[valid], cols[valid]
print("valid pixels:", len(rr))
block = np.stack([arrs[f][rr, cc] for f in FEATURES], axis=1)
if use_scaler:
    prob = final.predict_proba(scaler.transform(block))[:, 1]
else:
    prob = final.predict_proba(block)[:, 1]
prob_map = np.full((ny, nx), np.nan)
prob_map[rr, cc] = prob
with rasterio.open(os.path.join(OUT, "prob_map_final_v3.tif"), "w", driver="GTiff",
                   height=ny, width=nx, count=1, dtype="float32", crs="EPSG:4326",
                   transform=tf2, compress="lzw", nodata=np.nan) as dst:
    dst.write(prob_map.astype("float32"), 1)

p_vals = prob_map[valid]
q = np.nanquantile(p_vals, [0.2, 0.4, 0.6, 0.8])
classes = np.digitize(prob_map, q)
zones = classes[valid]
proba_sample = final.predict_proba(scaler.transform(X) if use_scaler else X)[:, 1]
sample_zone = np.digitize(proba_sample, q)
names = ["very_low", "low", "moderate", "high", "very_high"]
zstat = pd.DataFrame({"zone": names,
                      "area_frac": [zones[zones == k].size / len(zones) for k in range(5)]})
zstat["pos_frac"] = [((y[sample_zone == k] == 1).mean() if (sample_zone == k).sum() else np.nan) for k in range(5)]
zstat.to_csv(os.path.join(OUT, "zone_stats_final_v3.csv"), index=False, encoding="utf-8-sig")
print("\n=== ZONE STATS (11-factor, best:", best, ") ===")
print(zstat.to_string(index=False))
print("thresholds:", q)

with rasterio.open(os.path.join(OUT, "zones_final_v3.tif"), "w", driver="GTiff",
                   height=ny, width=nx, count=1, dtype="uint8", crs="EPSG:4326",
                   transform=tf2, compress="lzw", nodata=255) as dst:
    z_out = classes.astype("uint8"); z_out[~np.isfinite(prob_map)] = 255
    dst.write(z_out, 1)
print("Stage13 done")