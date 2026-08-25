"""
Stage 19 (P0-3): Temporal extrapolation validation.
Train XGBoost on 2008 coseismic landslides (as in main pipeline). Then for each
later inventory year (2013, 2015, 2017, 2019, 2020, 2021), compute the hit rate
(share of that year's NEW landslide pixels falling in predicted high/very-high
susceptibility zones) AND the capture rate (share of that year's polygons whose
representative pixel probability > threshold).

CRITICAL: exclude pixels overlapping the 2008 training polygons from the
later-year evaluation (otherwise spatial overlap inflates the metric).

Also reports: number of new polygons per year, fraction overlapping 2008 polygons.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xgboost as xgb
from shapely.geometry import box

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
LSDIR = r"C:\Users\Dell\Downloads\rsch1\experiment\data\landslide_extract\ls"
SEED = 42

FACTORS = ["elev", "slope", "aspect_sin", "aspect_cos", "plan_c", "prof_c",
           "twi", "spi", "sti", "relief", "ndvi"]
FEATURES = FACTORS + ["dist_stream"]
X0, X1, Y0, Y1 = 102.75, 103.65, 31.15, 31.85

# ---- load & prep samples identical to main pipeline ----
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
print("train samples:", X.shape)

model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                          random_state=SEED, n_jobs=-1)
model.fit(X, y)

# ---- load full-basin probability map (from main pipeline) ----
with rasterio.open(os.path.join(OUT, "prob_map_final_v3.tif")) as sp:
    prob = sp.read(1)
    tfp = sp.transform
with rasterio.open(os.path.join(OUT, "zones_final_v3.tif")) as sz:
    zones = sz.read(1)

# training (2008) polygons for overlap exclusion
g08 = gpd.read_file(os.path.join(LSDIR, "2008ls.shp"))
study_zone = box(X0, Y0, X1, Y1)
g08 = g08[g08.geometry.intersects(study_zone)]
print("2008 polygons in study area:", len(g08))

YEARS = ["2013ls", "2015ls", "2017ls", "2019ls", "2020ls", "2021ls"]
rows = []
for yr in YEARS:
    shp = os.path.join(LSDIR, yr + ".shp")
    if not os.path.exists(shp):
        print(f"skip {yr}: file missing")
        continue
    g = gpd.read_file(shp)
    g = g[g.geometry.intersects(study_zone)]
    total = len(g)
    # representative points
    pts = g.geometry.representative_point()
    px = ((pts.x - tfp.c) / tfp.a).astype(int)
    py = ((tfp.f - pts.y) / (-tfp.e)).astype(int)
    ok = (px >= 0) & (px < prob.shape[1]) & (py >= 0) & (py < prob.shape[0])
    px, py, g_ok = px[ok], py[ok], g[ok]

    # overlap with 2008 polygons (point-in-polygon test)
    ov = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts.x[ok], pts.y[ok]), crs="EPSG:4326") \
           .within(g08.geometry.unary_union)
    n_overlap = int(ov.sum())
    n_new = int(len(g_ok) - n_overlap)

    # hit/capture on NEW polygons (exclude overlap)
    new_mask = ~ov.values
    pnew = prob[py[new_mask], px[new_mask]]
    znew = zones[py[new_mask], px[new_mask]]
    hi_very = (znew >= 3)  # high or very high (zones 0-4)
    thr_quant = 0.2183     # 'high' threshold from main zoning
    cap = (pnew >= thr_quant)

    rows.append({
        "year": yr, "total_polygons": total, "in_study": len(g),
        "overlap_2008": n_overlap, "new_polygons": n_new,
        "new_valid_prob": int(len(pnew)),
        "hit_high_veryhigh_pct": round(100 * hi_very.mean(), 2) if len(hi_very) else np.nan,
        "capture_above_threshold_pct": round(100 * cap.mean(), 2) if len(cap) else np.nan,
    })
    print(f"{yr}: total={total} new(no-overlap)={n_new} "
          f"hit%={rows[-1]['hit_high_veryhigh_pct']} capture%={rows[-1]['capture_above_threshold_pct']}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "temporal_extrapolation.csv"), index=False, encoding="utf-8-sig")
print("\n=== temporal extrapolation summary ===")
print(res.to_string(index=False))
print("saved temporal_extrapolation.csv")