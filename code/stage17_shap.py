"""
Stage 17: SHAP explainability for the best model (XGBoost, 11 factors).
Produces: SHAP summary (bee swarm), mean |SHAP| bar, and top-2 dependence plots.
SCI manuscript Fig 7.
"""
import os
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
import shap
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
LABELS = {
    "elev": "Elevation", "slope": "Slope", "aspect_sin": "Aspect sin",
    "aspect_cos": "Aspect cos", "plan_c": "Plan curvature", "prof_c": "Profile curvature",
    "twi": "TWI", "spi": "SPI", "sti": "STI", "relief": "Relief",
    "ndvi": "NDVI", "dist_stream": "Dist. to streams"}

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
Xdf = df[FEATURES].copy()
y = df["label"].values
print("SHAP dataset:", Xdf.shape)

model = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                          random_state=SEED, n_jobs=-1)
model.fit(Xdf, y)

# SHAP with subsample for speed (3000 samples)
X_small = Xdf.sample(3000, random_state=SEED)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_small)

# 1) summary plot (bee swarm) - use display labels
X_disp = X_small.rename(columns=LABELS)
shap_values_disp = shap_values.copy()
# rename columns order must match; create DataFrame of shap values with display names
shap_df = pd.DataFrame(shap_values, columns=[LABELS[f] for f in FEATURES])
# rename X_disp columns to match display names again (they already are)

fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(shap_df.values, X_disp.values, feature_names=list(X_disp.columns),
                  show=False, max_display=12)
plt.title("SHAP Summary (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig7_shap_summary.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig7_shap_summary.png saved")

# 2) mean |SHAP| bar
mean_abs = np.abs(shap_df).mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(mean_abs.index, mean_abs.values, color="#0072B2")
ax.set_xlabel("mean |SHAP|")
ax.set_title("Feature Importance by SHAP (XGBoost)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_shap_bar.png"), dpi=300)
plt.close(fig)
print("fig7_shap_bar.png saved")

# 3) dependence plots for top-2 factors
top2 = mean_abs.index[-2:].tolist()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, feat in zip(axes, top2):
    idx = list(X_disp.columns).index(feat)
    shap.dependence_plot(idx, shap_df.values, X_disp.values,
                         feature_names=list(X_disp.columns), ax=ax, show=False)
    ax.set_title(f"SHAP dependence: {feat}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_shap_dependence.png"), dpi=300)
plt.close(fig)
print("fig7_shap_dependence.png saved")

# save numeric summary
mean_abs.to_csv(os.path.join(OUT, "shap_mean_abs.csv"), encoding="utf-8-sig")
print("\n=== mean |SHAP| (desc) ===")
print(mean_abs.sort_values(ascending=False).round(4).to_string())
print("Stage17 done")