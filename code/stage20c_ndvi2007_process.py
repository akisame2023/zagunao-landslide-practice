"""
Stage 20c (P0-4): process 3 pre-quake 2007 NDVI tiles -> ndvi_2007.tif (30m).
Uses QGIS GDAL for HDF4 -> median composite -> resample to factor grid.
"""
import os
import subprocess
import numpy as np
import rasterio

QGIS_BIN = r"C:\Program Files\QGIS 3.40.14\bin"
GDAL_DATA = r"C:\Program Files\QGIS 3.40.14\share\gdal"
os.environ["GDAL_DATA"] = GDAL_DATA
os.environ["PATH"] = QGIS_BIN + os.pathsep + os.environ["PATH"]

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
DATADIR = r"C:\Users\Dell\Downloads\rsch1\experiment\data\modis2007"
TMP = os.path.join(OUT, "ndvi2007_tmp")
os.makedirs(TMP, exist_ok=True)
X0, Y0, X1, Y1 = 102.75, 31.15, 103.65, 31.85

SCENES = [
    "MOD13Q1.A2007177.h26v05.061.2021068095141",
    "MOD13Q1.A2007193.h26v05.061.2021068163334",
    "MOD13Q1.A2007241.h26v05.061.2021073200334",
]

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CMD FAIL:", " ".join(cmd[:3]), "...", r.stderr[-300:])
        return False
    return True

tifs = []
for sid in SCENES:
    hdf = os.path.join(DATADIR, sid + ".hdf")
    if not os.path.exists(hdf) or os.path.getsize(hdf) < 200_000_000:
        print("skip", sid, os.path.getsize(hdf) if os.path.exists(hdf) else "missing")
        continue
    sub_path = f'HDF4_EOS:EOS_GRID:"{hdf}":MODIS_Grid_16DAY_250m_500m_VI:"250m 16 days NDVI"'
    raw = os.path.join(TMP, sid + "_raw.tif")
    geo = os.path.join(TMP, sid + "_geo.tif")
    if not run(["gdal_translate", "-q", sub_path, raw]):
        continue
    if not run(["gdalwarp", "-q", "-t_srs", "EPSG:4326", "-te", str(X0), str(Y0), str(X1), str(Y1),
                "-tr", "0.0025", "0.0025", "-r", "bilinear", "-overwrite", raw, geo]):
        continue
    tifs.append(geo)
print("processed tiles:", len(tifs), tifs)

arrs = []
for geo in tifs:
    with rasterio.open(geo) as s:
        a = s.read(1).astype(np.float64)
    a[a <= -3000] = np.nan
    a[a > 10000] = np.nan
    arrs.append(a)
stack = np.stack(arrs)
ndvi07 = np.nanmedian(stack, axis=0) / 10000.0
print("2007 median NDVI stats:", np.nanmin(ndvi07), np.nanmean(ndvi07), np.nanmax(ndvi07))

with rasterio.open(tifs[0]) as s0:
    prof = s0.profile; h0, w0 = s0.height, s0.width; tf0 = s0.transform
med_tif = os.path.join(TMP, "ndvi2007_median250.tif")
with rasterio.open(med_tif, "w", driver="GTiff", height=h0, width=w0, count=1,
                   dtype="float32", crs="EPSG:4326", transform=tf0) as dst:
    dst.write(ndvi07.astype("float32"), 1)
with rasterio.open(os.path.join(OUT, "elev.tif")) as s0:
    ny30, nx30 = s0.shape
out30 = os.path.join(OUT, "ndvi_2007.tif")
run(["gdalwarp", "-q", "-te", str(X0), str(Y0), str(X1), str(Y1),
     "-ts", str(nx30), str(ny30), "-r", "bilinear", "-overwrite", med_tif, out30])
with rasterio.open(out30) as s:
    a = s.read(1)
    print("ndvi_2007.tif:", a.shape, "valid:", np.isfinite(a).sum(),
          "mean:", np.nanmean(a).round(3), "p5-p95:", np.nanpercentile(a, [5, 95]).round(3))
print("P0-4 NDVI2007 done")