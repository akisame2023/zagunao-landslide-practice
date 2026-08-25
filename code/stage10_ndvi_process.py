"""
Stage 10: Process 5 MODIS NDVI tiles -> summer median NDVI -> 30m aligned GeoTIFF.
Uses QGIS bundled GDAL (verified HDF4 support).
Pipeline:
1. gdal_translate: HDF NDVI subdataset -> GeoTIFF (sinusoidal, DN*0.0001)
2. gdalwarp: reproject to EPSG:4326, clip to study area bounds
3. numpy median composite over scenes -> ndvi.tif on factor grid (bilinear resample 30m)
"""
import os
import subprocess
import numpy as np
import rasterio

QGIS_BIN = r"C:\Program Files\QGIS 3.40.14\bin"
GDAL_DATA = r"C:\Program Files\QGIS 3.40.14\share\gdal"
os.environ["GDAL_DATA"] = GDAL_DATA
os.environ["PATH"] = QGIS_BIN + os.pathsep + os.environ["PATH"]

MOD = r"C:\Users\Dell\Downloads\rsch1\experiment\data\modis"
OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
TMP = os.path.join(OUT, "ndvi_tmp")
os.makedirs(TMP, exist_ok=True)

SCENES = [
    "MOD13Q1.A2020209.h26v05.061.2020342021543",
    "MOD13Q1.A2020225.h26v05.061.2020344204509",
    "MOD13Q1.A2020241.h26v05.061.2020346093237",
    "MOD13Q1.A2020257.h26v05.061.2020347062509",
    "MOD13Q1.A2020273.h26v05.061.2020349200629",
]
# study area (matches factor rasters)
X0, Y0, X1, Y1 = 102.75, 31.15, 103.65, 31.85

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CMD FAIL:", " ".join(cmd[:4]), "...")
        print(r.stderr[-500:])
        return False
    return True

tifs = []
for sid in SCENES:
    hdf = os.path.join(MOD, sid + ".hdf")
    if not os.path.exists(hdf) or os.path.getsize(hdf) < 200_000_000:
        print(f"SKIP incomplete: {sid} ({os.path.getsize(hdf) if os.path.exists(hdf) else 'missing'})")
        continue
    sub_path = f'HDF4_EOS:EOS_GRID:"{hdf}":MODIS_Grid_16DAY_250m_500m_VI:"250m 16 days NDVI"'
    raw = os.path.join(TMP, sid + "_ndvi_raw.tif")
    # translate to tiff (retain DN; scale later in numpy)
    if not run(["gdal_translate", "-q", sub_path, raw]):
        continue
    # reproject sinusoidal -> EPSG:4326, clip to bbox, keep DN
    geo = os.path.join(TMP, sid + "_ndvi_geo.tif")
    if not run(["gdalwarp", "-q", "-t_srs", "EPSG:4326", "-te", str(X0), str(Y0), str(X1), str(Y1),
                "-tr", "0.0025", "0.0025", "-r", "bilinear", "-overwrite", raw, geo]):
        continue
    tifs.append((sid, geo))
    print("ok:", sid)

print("tiles ready:", len(tifs))

# median composite
arrs = []
with rasterio.open(tifs[0][1]) as s0:
    profile = s0.profile
    height, width = s0.height, s0.width
    tf0 = s0.transform
for sid, geo in tifs:
    with rasterio.open(geo) as s:
        a = s.read(1).astype(np.float64)
    a[a <= -3000] = np.nan   # fill
    a[a > 10000] = np.nan    # snow/cloud mask (MODIS fill flags)
    arrs.append(a)
stack = np.stack(arrs)
ndvi250 = np.nanmedian(stack, axis=0) / 10000.0  # scale factor
print("250m median NDVI stats:", np.nanmin(ndvi250), np.nanmean(ndvi250), np.nanmax(ndvi250))

# --- resample to 30m factor grid ---
with rasterio.open(os.path.join(OUT, "elev.tif")) as s0:
    tf30 = s0.transform
    ny30, nx30 = s0.shape
# use gdalwarp from 250m geo to exact 30m grid
med_tif = os.path.join(TMP, "ndvi_median250.tif")
with rasterio.open(med_tif, "w", driver="GTiff", height=height, width=width,
                   count=1, dtype="float32", crs="EPSG:4326", transform=tf0) as dst:
    dst.write(ndvi250.astype("float32"), 1)
out30 = os.path.join(OUT, "ndvi.tif")
if not run(["gdalwarp", "-q", "-te", str(X0), str(Y0), str(X1), str(Y1),
            "-ts", str(nx30), str(ny30), "-r", "bilinear", "-overwrite",
            med_tif, out30]):
    print("30m warp failed")
else:
    with rasterio.open(out30) as s:
        a = s.read(1)
        print("FINAL ndvi.tif:", a.shape, "valid:", np.isfinite(a).sum(),
              "mean:", np.nanmean(a).round(3), "min/max:", np.nanmin(a).round(3), np.nanmax(a).round(3))
    print("NDVI_DONE")