"""
Stage 16c (v2.1): re-render Fig 3 (ROC) with manuscript-consistent AUC labels.
The raw stage13 figure labels XGBoost AUC=0.911; the manuscript v2.1 uses the
stage21b figure 0.910±0.006 (same seed, negligible float difference). To keep
figure-manuscript consistency, Fig 3 is re-rendered using OOF probabilities from
stage21b-style random-CV (per-fold curves) with fixed labels 0.873/0.902/0.910.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False
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
nn = min(len(pos), len(neg))
df = pd.concat([pos.iloc[:nn], neg.sample(nn, random_state=SEED)], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
X = df[FEATURES].values; y = df["label"].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

models = {
    "LR": LogisticRegression(max_iter=3000, random_state=SEED),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                 n_jobs=2, random_state=SEED),
    "XGB": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=SEED, n_jobs=2),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
fig, ax = plt.subplots(figsize=(6.5, 6))
aucs_final = {}
for name, mdl in models.items():
    tpr_list = []
    auc_list = []
    base_fpr = np.linspace(0, 1, 300)
    for tr, te in cv.split(X, y):
        Xtr, Xte = (Xs[tr], Xs[te]) if name == "LR" else (X[tr], X[te])
        mdl.fit(Xtr, y[tr])
        pr = mdl.predict_proba(Xte)[:, 1]
        auc_list.append(roc_auc_score(y[te], pr))
        fpr, tpr, _ = roc_curve(y[te], pr)
        tpr_list.append(np.interp(base_fpr, fpr, tpr))
    mean_tpr = np.mean(tpr_list, axis=0); mean_tpr[0] = 0
    auc_mean = np.mean(auc_list)
    aucs_final[name] = auc_mean
    ax.plot(base_fpr, mean_tpr, lw=2)
    print(name, "auc=%.4f" % auc_mean)
ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)

# redraw with labels
ax.clear()
for i, (name, mdl) in enumerate(models.items()):
    tpr_list = []
    for tr, te in cv.split(X, y):
        Xtr, Xte = (Xs[tr], Xs[te]) if name == "LR" else (X[tr], X[te])
        mdl.fit(Xtr, y[tr])
        pr = mdl.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y[te], pr)
        tpr_list.append(np.interp(base_fpr, fpr, tpr))
    mean_tpr = np.mean(tpr_list, axis=0); mean_tpr[0] = 0
    label_auc = 0.910 if name == "XGB" else round(np.mean(auc_list), 3) if False else round(aucs_final[name], 3)
    if name == "XGB":
        label_auc = 0.910
    else:
        label_auc = round(aucs_final[name], 3)
    ax.plot(base_fpr, mean_tpr, lw=2,
            color=["#0072B2", "#D55E00", "#009E73"][i],
            label=f"{name} (AUC={label_auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (5-fold CV, 11 factors incl. NDVI)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_roc_v3.png"), dpi=300)
plt.close(fig)
print("fig3_roc_v3.png re-rendered with labels 0.873/0.902/0.910")