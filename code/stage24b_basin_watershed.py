"""
Stage 24b (P0-2 final): basin approximation via scipy watershed (no pysheds/numba).
Method: invert DEM; use stream network pixels (distance-to-stream == 0, from the
existing dist_stream_low.tif) as markers per outlet; watershed on the inverted DEM
segments the landscape into catchments; keep the catchment containing the Zagunao
outlet cell (max flow-accum proxy via elevation minima near Wenchuan).

This is a defensible approximation (documented in the paper as such); the exact
hydro-mask is Supplementary and not asserted as an authoritative basin boundary.
"""
import os
import numpy as np
import rasterio
from scipy import ndimage

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"

with rasterio.open(os.path.join(OUT, "elev.tif")) as s0:
    dem = s0.read(1).astype(np.float64)
    tf30 = s0.transform
    ny, nx = s0.shape
    crs = s0.crs
dem[dem < -1000] = np.nan

# 1) stream network mask from existing distance raster (dist==0 -> channel)
with rasterio.open(os.path.join(OUT, "dist_stream_low.tif")) as s1:
    dist = s1.read(1).astype(np.float64)
    tf_low = s1.transform
# upsample channel mask to 30m grid (nearest)
from rasterio.enums import Resampling
with rasterio.open(os.path.join(OUT, "dist_stream_low.tif")) as s1:
    dist30 = s1.read(1, out_shape=(ny, nx), resampling=Resampling.nearest)
channel = dist30 <= 0.01
print("channel cells:", int(channel.sum()))

# 2) inverted DEM for watershed (lower = more upstream); fill nan with high
inv = np.where(np.isfinite(dem), -dem, 1e9)

# 3) markers: label each channel segment as separate catchment seed via
#    connected components on the channel mask
lbl, ncomp = ndimage.label(channel, structure=np.ones((3, 3)))
print("channel components:", ncomp)

# 4) watershed
from scipy.ndimage import watershed_ift
# watershed_ift needs integer image: quantize inverted DEM
inv8 = np.clip(inv, -5000, 5000)
inv_int = ((inv8 - inv8.min()) / (inv8.max() - inv8.min()) * 32767).astype(np.uint16)
catch_lbl = watershed_ift(inv_int, lbl, structure=np.ones((3, 3)))

# 5) choose the catchment containing the Zagunao outlet near Wenchuan (103.59E, 31.48N)
outlet_c = int(round((103.59 - tf30.c) / tf30.a))
outlet_r = int(round((tf30.f - 31.48) / (-tf30.e)))
outlet_lbl = catch_lbl[outlet_r, outlet_c]
print("outlet label at (103.59E,31.48N):", outlet_lbl)
basin = (catch_lbl == outlet_lbl).astype(np.uint8)
# remove nans
basin[~np.isfinite(dem)] = 0
n = int(basin.sum())
print("basin cells:", n, "= area km2:", round(n * 30 * 30 / 1e6, 1),
      "(fraction of window:", round(100 * n / (ny * nx), 1), "%)")

# sanity: basin lon/lat span
ys, xs = np.where(basin > 0)
print("basin lon:", round(tf30.c + xs.min() * tf30.a, 3), "-",
      round(tf30.c + xs.max() * tf30.a, 3),
      "| lat:", round(tf30.f - ys.max() * (-tf30.e), 3), "-",
      round(tf30.f - ys.min() * (-tf30.e), 3))

with rasterio.open(os.path.join(OUT, "basin_mask.tif"), "w", driver="GTiff",
                   height=ny, width=nx, count=1, dtype="uint8",
                   crs=crs, transform=tf30, compress="lzw", nodata=0) as dst:
    dst.write(basin, 1)

# 6) zoning stats within basin (window-defined zones from zones_final_v3.tif)
with rasterio.open(os.path.join(OUT, "zones_final_v3.tif")) as sz:
    zones = sz.read(1)
with rasterio.open(os.path.join(OUT, "prob_map_final_v3.tif")) as sp:
    prob = sp.read(1)

mask = (basin > 0) & np.isfinite(prob)
import pandas as pd
names = ["very_low", "low", "moderate", "high", "very_high"]
rows = []
for k in range(5):
    zk = mask & (zones == k)
    rows.append({"zone": names[k], "basin_area_share_pct": round(100 * zk.sum() / max(mask.sum(), 1), 2),
                 "n_pixels": int(zk.sum())})
zstat = pd.DataFrame(rows)
zstat.to_csv(os.path.join(OUT, "zoning_basin_mask.csv"), index=False, encoding="utf-8-sig")

# landslide samples inside basin
import pandas as pd
samples = pd.read_csv(os.path.join(OUT, "samples_raw2.csv"))
lon = samples["lon"].values; lat = samples["lat"].values
c = np.round((lon - tf30.c) / tf30.a).astype(int)
r = np.round((tf30.f - lat) / (-tf30.e)).astype(int)
ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
inb = basin[r[ok], c[ok]] > 0
lbls = samples["label"].values[ok]
print("positives in basin:", int(((lbls == 1) & inb).sum()), "/", int((lbls == 1).sum()))
print("negatives in basin:", int(((lbls == 0) & inb).sum()), "/", int((lbls == 0).sum()))
print("\nbasin zoning table:")
print(zstat.to_string(index=False))
print("saved basin_mask.tif + zoning_basin_mask.csv")