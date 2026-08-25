"""
Stage 2b (rigorous): grid-based negative sampling.
Rasterize landslide polygons into a mask, dilate by N cells, sample negatives
only from allowed (non-landslide, non-buffer) cells. Positive = landslide cells (1 per polygon centroid).
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.ndimage import binary_dilation

DATA = r"C:\Users\Dell\Downloads\rsch1\experiment\data"
OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
LSDIR = os.path.join(DATA, "landslide_extract", "ls")
SEED = 42
BUFFER_CELLS = 2  # ~60m exclusion buffer around landslides

rng = np.random.default_rng(SEED)

# reference raster: elevation
with rasterio.open(os.path.join(OUT, "elev.tif")) as src:
    elev = src.read(1)
    tf = src.transform
    ny, nx = elev.shape
    valid = ~np.isnan(elev)  # wait: elev tif written as float32 with nan nodata?
    valid = elev > -1000

# clip bounds of raster
bounds = rasterio.transform.array_bounds(ny, nx, tf)

g = gpd.read_file(os.path.join(LSDIR, "2008ls.shp"))
zone = box(*bounds)
g = g[g.geometry.intersects(zone)].copy()
print("landslides:", len(g))

# rasterize polygons -> 1 where landslide
shapes = [(geom, 1) for geom in g.geometry if not geom.is_empty]
mask = rasterize(shapes, out_shape=(ny, nx), transform=tf,
                 fill=0, dtype="uint8", all_touched=True)
# also rasterize an ALLOWED-zone = valid DEM cells (elevation defined)
allowed = valid & (mask == 0)
# dilate landslide mask by buffer
buffered = binary_dilation(mask == 1, iterations=BUFFER_CELLS)
allowed &= ~buffered
print("allowed cells:", allowed.sum(), "landslide cells:", (mask == 1).sum(),
      "valid cells:", valid.sum())

# sample negatives from allowed cells (no duplicates), count = number of landslide polygons
n_pos = len(g)
ys, xs = np.nonzero(allowed)
idx = rng.choice(len(ys), size=min(n_pos, len(ys)), replace=False)
neg_rows = ys[idx]; neg_cols = xs[idx]
# to lon/lat
lon = tf.c + (neg_cols + 0.5) * tf.a
lat = tf.f - (neg_rows + 0.5) * (-tf.e)
neg_df = pd.DataFrame({"label": 0, "lon": lon, "lat": lat})
print("negatives:", len(neg_df))

# positives: representative point -> pixel row/col -> retain
pts = g.geometry.representative_point()
px = (pts.x - tf.c) / tf.a
py = (tf.f - pts.y) / (-tf.e)
pos_rows = np.round(py).astype(int)
pos_cols = np.round(px).astype(int)
valid_pos = (pos_rows >= 0) & (pos_rows < ny) & (pos_cols >= 0) & (pos_cols < nx)
pos_df = pd.DataFrame({
    "label": 1,
    "lon": pts.x.values[valid_pos],
    "lat": pts.y.values[valid_pos],
})
print("positives:", len(pos_df), "(dropped", (n_pos - len(pos_df)), "outside raster)")

samples = pd.concat([pos_df, neg_df], ignore_index=True)
print("balance:", samples["label"].value_counts().to_dict())
samples.to_csv(os.path.join(OUT, "samples_raw2.csv"), index=False)
print("Stage2b samples_raw2.csv written:", len(samples))