"""
Stage 26: re-render Fig 2 (study area) with clarified inventory-count annotation.
Title: "研究区地形与 2008 年滑坡分布（N = 13,288，编目多边形数；去重缺失剔除后建模样本 13,086）"
Same map as stage7 but with the inventory-vs-model-sample annotation to remove
the N=13,288 vs 13,086 ambiguity flagged in the figure review.
"""
import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False
OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"

with rasterio.open(os.path.join(OUT, "dem_clip.tif")) as src:
    dem = src.read(1); tfd = src.transform

g = gpd.read_file(r"C:\Users\Dell\Downloads\rsch1\experiment\data\landslide_extract\ls\2008ls.shp")
bounds = rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], tfd)
g = g[g.geometry.intersects(box(*bounds))]
n_inv = len(g)   # inventory polygons in window = 13,288
print("inventory polygons in window:", n_inv)

fig, ax = plt.subplots(figsize=(8.5, 7))
ax.imshow(np.log1p(np.clip(dem, 0, None)), cmap="terrain", interpolation="nearest",
          extent=[tfd.c, tfd.c + tfd.a * dem.shape[1], tfd.f + tfd.e * dem.shape[0], tfd.f])
g.plot(ax=ax, color="#D55E00", markersize=0.25, alpha=0.55)
ax.set_title(f"研究区地形与 2008 年滑坡分布（N = {n_inv:,}，编目多边形；建模正样本 13,086）")
ax.set_xlabel("经度 / °E"); ax.set_ylabel("纬度 / °N")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_study_area.png"), dpi=250)
plt.close(fig)
print("fig2_study_area.png re-rendered (v2, annotated)")