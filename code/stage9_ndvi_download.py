"""
Stage 9 (optional but recommended): MODIS NDVI download + processing.
- Downloads 4-6 summer MOD13Q1 tiles (h26v05, covers 102.75-103.65E/31.15-31.85N) for 2020
- Uses NASA Earthdata Bearer token (created at https://lpdaac.usgs.gov/user/tokens)
- Extracts NDVI band (250m, scale 0.0001), merges, clips to study area, resamples to 30m
- Writes ndvi.tif aligned with existing factor rasters -> ready for stage8 rerun with NDVI factor

Usage: python stage9_ndvi.py <YOUR_EARTHDATA_TOKEN>
Token is a session credential; delete it at lpdaac.usgs.gov/user/tokens afterwards.
"""
import sys
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import zoom
from datetime import datetime

TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EARTHDATA_TOKEN", "")
if not TOKEN:
    sys.exit("No token. Pass as argv[1] or set EARTHDATA_TOKEN. Create at https://lpdaac.usgs.gov/user/tokens")

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
DATADIR = r"C:\Users\Dell\Downloads\rsch1\experiment\data\modis"
os.makedirs(DATADIR, exist_ok=True)

# summer 2020 MOD13Q1 scenes for tile h26v05 (already probed in this session)
SCENES = [
    "MOD13Q1.A2020209.h26v05.061.2020342021543",
    "MOD13Q1.A2020225.h26v05.061.2020344204509",
    "MOD13Q1.A2020241.h26v05.061.2020346093237",
    "MOD13Q1.A2020257.h26v05.061.2020347062509",
    "MOD13Q1.A2020273.h26v05.061.2020349200629",
]
BASE = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13Q1.061"
HDRS = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "Mozilla/5.0"}

import urllib.request

# ---- 1. download scenes (skip if already present & >1MB) ----
paths = []
for sid in SCENES:
    dest = os.path.join(DATADIR, sid + ".hdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"skip (exists): {sid}")
        paths.append(dest)
        continue
    url = f"{BASE}/{sid}/{sid}.hdf"
    req = urllib.request.Request(url, headers=HDRS)
    try:
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk); total += len(chunk)
        print(f"downloaded {sid}: {total} bytes")
        if total < 100_000:  # likely an HTML error page
            print("  !! possible auth failure / HTML error page")
        paths.append(dest)
    except Exception as e:
        print(f"FAIL {sid}: {type(e).__name__}: {str(e)[:100]}")

if not paths:
    sys.exit("no scenes downloaded - check token")

# ---- 2. extract NDVI from HDF (SDS '250m 16 days NDVI') ----
try:
    import h5py
    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False

def read_ndvi_hdf(path):
    """Return (ndvi array 4800x4800, scale factor). Uses rasterio's HDF driver first."""
    with rasterio.open(f"HDF4_EOS:EOS_GRID:{path}:MOD_Grid_250m_2D:250m 16 days NDVI") as src:
        arr = src.read(1)
        scale = 0.0001
        return arr, scale

ndvis = []
for p in paths:
    try:
        arr, scale = read_ndvi_hdf(p)
        ndvis.append(arr.astype(np.float64) * scale)
        print(f"extracted NDVI from {os.path.basename(p)}: shape={arr.shape}")
    except Exception as e:
        print(f"extract FAIL {os.path.basename(p)}: {type(e).__name__}: {str(e)[:150]}")

if not ndvis:
    sys.exit("no NDVI extracted - check HDF driver (rasterio HDF4 requires GDAL HDF4 build)")

# ---- 3. summer composite (median) ----
ndvi_median = np.median(np.stack(ndvis), axis=0)  # 4800x4800, 250m
# valid range
ndvi_median[(ndvi_median < -0.2) | (ndvi_median > 1.0)] = np.nan
print("NDVI 250m stats:", np.nanmin(ndvi_median), np.nanmean(ndvi_median), np.nanmax(ndvi_median))

# ---- 4. clip to study area (102.75-103.65E, 31.15-31.85N) with MODIS sinusoidal->EPSG:4326 ----
# MOD13Q1 h26v05 in sinusoidal; do a simple geographic approximation:
# read geotransform from HDF metadata via rasterio
with rasterio.open(f"HDF4_EOS:EOS_GRID:{paths[0]}:MOD_Grid_250m_2D:250m 16 days NDVI") as src:
    gridd_tr = src.transform
    gridd_crs = src.crs
print("HDF crs:", gridd_crs, "transform:", gridd_tr)

# Reproject to EPSG:4326 using rasterio WarpedVRT (needs network-free on-machine reprojection)
from rasterio.warp import calculate_default_transform, reproject, Resampling
ny0, nx0 = ndvi_median.shape
dst_crs = "EPSG:4326"
# region of interest approx for h26v05 in lon/lat: use generous bbox
bbox = (100.0, 29.0, 106.0, 34.0)  # lonmin, latmin, lonmax, latmax (generous)
dst_transform, dst_w, dst_h = calculate_default_transform(
    gridd_crs, dst_crs, nx0, ny0, *bbox if gridd_crs != dst_crs else (0,0,nx0,ny0), resolution=0.0025)
# simpler: reproject whole tile to 4326 at 0.0025 deg (~250m)
ndvi_4326 = np.full((dst_h, dst_w), np.nan, dtype=np.float64)
reproject(ndvi_median, ndvi_4326,
          src_transform=gridd_tr, src_crs=gridd_crs,
          dst_transform=dst_transform, dst_crs=dst_crs,
          resampling=Resampling.bilinear)

# ---- 5. clip & resample to 30m factor grid ----
with rasterio.open(os.path.join(OUT, "elev.tif")) as s0:
    tf30 = s0.transform
    ny30, nx30 = s0.shape
win = from_bounds(102.75, 31.15, 103.65, 31.85, dst_transform)
win = win.intersection(rasterio.coords.bounds_to_window(
    rasterio.transform.array_bounds(dst_h, dst_w, dst_transform), dst_transform))
ndvi_clip = ndvi_4326[int(win.row_off):int(win.row_off + win.height),
                      int(win.col_off):int(win.col_off + win.width)]
# resample clip to 30m grid via zoom (approx; use reproject for publication quality)
zoom_y = ny30 / ndvi_clip.shape[0]
zoom_x = nx30 / ndvi_clip.shape[1]
ndvi_30 = zoom(ndvi_clip, (zoom_y, zoom_x), order=1)

with rasterio.open(os.path.join(OUT, "ndvi.tif"), "w", driver="GTiff",
                   height=ny30, width=nx30, count=1, dtype="float32",
                   crs="EPSG:4326", transform=tf30, compress="lzw", nodata=np.nan) as dst:
    dst.write(ndvi_30.astype("float32"), 1)
print("WROTE ndvi.tif (30m):", ndvi_30.shape, "valid frac:",
      np.isfinite(ndvi_30).mean().round(3))
print("NDVI 30m stats:", np.nanmin(ndvi_30).round(3), np.nanmean(ndvi_30).round(3), np.nanmax(ndvi_30).round(3))
print("Stage9 NDVI done. Next: add 'ndvi' to FEATURES in stage8_final.py and rerun.")