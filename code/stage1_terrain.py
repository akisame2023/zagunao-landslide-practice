"""
Stage 1: Mosaic DEM tiles, clip to study area, derive terrain factors.
Study area (Zagunao River Basin, Wenchuan): ~102.8-103.6E, 31.2-31.8N
Factors derived here: elevation, slope, aspect, plan curvature, profile curvature, TWI, dist-to-stream
"""
import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from scipy import ndimage

DATA = r"C:\Users\Dell\Downloads\rsch1\experiment\data"
OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
os.makedirs(OUT, exist_ok=True)

# --- 1. Merge 4 DEM tiles ---
tiles = [os.path.join(DATA, "dem", t) for t in
         ["N31E102.tif", "N31E103.tif", "N32E102.tif", "N32E103.tif"]]
srcs = [rasterio.open(t) for t in tiles]
mosaic, transform = merge(srcs)
merged_path = os.path.join(OUT, "dem_merged.tif")
with rasterio.open(merged_path, "w", driver="GTiff", height=mosaic.shape[1],
                   width=mosaic.shape[2], count=1, dtype=mosaic.dtype,
                   crs="EPSG:4326", transform=transform, compress="lzw") as dst:
    dst.write(mosaic[0], 1)
for s in srcs:
    s.close()
print("merged:", mosaic.shape, "res:", transform.a)

# --- 2. Study area clip (about 102.8-103.6E, 31.2-31.8N) ---
with rasterio.open(merged_path) as src:
    # window for bounds
    from rasterio.windows import from_bounds
    w = from_bounds(102.75, 31.15, 103.65, 31.85, src.transform)
    w = w.intersection(src.window(*src.bounds))
    dem = src.read(1, window=w)
    tf = src.window_transform(w)
    meta = src.meta.copy()
    meta.update(height=dem.shape[0], width=dem.shape[1], transform=tf)
    clip_path = os.path.join(OUT, "dem_clip.tif")
    with rasterio.open(clip_path, "w", **meta) as dst:
        dst.write(dem, 1)
print("clip:", dem.shape, "bounds:", rasterio.coords.BoundingBox(*rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], tf)))

# --- 3. Terrain factors (neighborhood ~3x3 at 0.00028 deg ~ 30m) ---
dem = dem.astype(np.float64)
dem[dem < -1000] = np.nan  # nodata

# convert gradient from deg to meters: 1 deg lat ~ 111320 m; 1 deg lon ~ 111320*cos(lat) m
lat_mid = np.deg2rad((tf.f + tf.f + tf.e * dem.shape[0]) / 2)
m_per_deg_x = 111320.0 * np.cos(lat_mid)
m_per_deg_y = 111320.0

gy_deg, gx_deg = np.gradient(dem)  # dz per cell (np.gradient uses unit spacing)
m_per_cell_x = m_per_deg_x * tf.a   # meters per cell in x
m_per_cell_y = m_per_deg_y * (-tf.e)
gx = gx_deg / m_per_cell_x          # dz per meter
gy = gy_deg / m_per_cell_y
slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
aspect = np.degrees(np.arctan2(-gx, gy)) % 360.0

# curvatures via second derivatives (Horn's method simplified), all in meter space
dxx_deg = np.gradient(gx_deg, axis=1) / (m_per_cell_x ** 2)
dyy_deg = np.gradient(gy_deg, axis=0) / (m_per_cell_y ** 2)
dxy_deg = np.gradient(gx_deg, axis=0) / (m_per_cell_x * m_per_cell_y)
plan_c = -(dxx_deg * gy**2 - 2 * dxy_deg * gx * gy + dyy_deg * gx**2) / ((gx**2 + gy**2) ** 1.5 + 1e-9)
prof_c = -(dxx_deg * gx**2 + 2 * dxy_deg * gx * gy + dyy_deg * gy**2) / ((gx**2 + gy**2) * np.sqrt(1 + gx**2 + gy**2) + 1e-9)

# TWI = ln(a/tan b), a = specific catchment area (D8 flow accumulation)
# use simple flow accumulation via D8 (implemented below)
from collections import deque

def d8_accum(dem):
    """Standard D8 flow accumulation on a grid (returns accumulation count)."""
    ny, nx = dem.shape
    accum = np.zeros((ny, nx), dtype=np.float64)
    # dirs: 8 neighbors
    dr = [-1,-1,-1,0,0,1,1,1]
    dc = [-1,0,1,-1,1,-1,0,1]
    # deterministic priority by max drop
    order = []
    stack = np.argsort(dem[~np.isnan(dem)].ravel())  # not used
    # simple O(N^2) approach too slow for big grid; use priority queue via ndimage-less method:
    # Use the classic "flow direction by steepest descent" with iterative accumulation.
    flen = np.zeros((ny, nx), dtype=np.float64)
    # precompute steepest neighbor
    dirs = np.zeros((ny, nx), dtype=np.int8)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if np.isnan(dem[i,j]):
                dirs[i,j] = -1
                continue
            best = 0.0; bi = -1
            for k in range(8):
                ni, nj = i+dr[k], j+dc[k]
                if np.isnan(dem[ni,nj]):
                    continue
                drop = dem[i,j] - dem[ni,nj]
                if drop > best:
                    best = drop; bi = k
            dirs[i,j] = bi
    # accumulate by processing cells in descending elevation (topological order)
    order = np.dstack(np.unravel_index(np.argsort(-dem.ravel(), kind="stable"), dem.shape))[0]
    for (i, j) in order:
        if dirs[i,j] < 0:
            continue
        k = dirs[i,j]
        ni, nj = i+dr[k], j+dc[k]
        if 0 <= ni < ny and 0 <= nj < nx and dirs[ni,nj] >= 0:
            flen[ni,nj] += flen[i,j] + 1.0
    return flen

# D8 on full 3600x3600-ish clip may be slow in pure python; use coarser fallback:
# Downsample by factor 4 (~120m) for TWI to keep runtime sane
ds = 4
dem_low = dem[::ds, ::ds]
dem_low = ndimage.gaussian_filter(dem_low, 1)
fa_low = d8_accum(dem_low)
# specific catchment area (cells) * cell area
cell_m2 = (m_per_cell_x * ds) * (m_per_cell_y * ds)
sca = (fa_low + 1.0) * cell_m2
gy_l, gx_l = np.gradient(dem_low)  # per low-res cell
gx_lm = gx_l / (ds * m_per_cell_x)
gy_lm = gy_l / (ds * m_per_cell_y)
slope_l = np.arctan(np.sqrt(gx_lm**2 + gy_lm**2))
twi_low = np.log(sca / (np.tan(slope_l) + 1e-9))
twi_low[(twi_low < -10) | (twi_low > 30)] = np.nan

# Streams: threshold flow accumulation (low-res) -> distance to stream (in low-res pixels, resampled)
stream_mask = fa_low > 1000  # channels
from scipy.ndimage import distance_transform_edt
dist_stream_low = distance_transform_edt(~stream_mask) * (ds * m_per_cell_y)

def write_geotiff(path, arr, tf, meta0, transform_ok=True):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=tf, compress="lzw", nodata=np.nan) as dst:
        dst.write(arr.astype("float32"), 1)

tf_low = rasterio.transform.from_origin(tf.c, tf.f, tf.a*ds, -tf.e*ds)
write_geotiff(os.path.join(OUT, "twi.tif"), twi_low, tf_low, meta)
write_geotiff(os.path.join(OUT, "dist_stream_low.tif"), dist_stream_low, tf_low, meta)

# Store full-res terrain factors
write_geotiff(os.path.join(OUT, "elev.tif"), dem, tf, meta)
write_geotiff(os.path.join(OUT, "slope.tif"), slope, tf, meta)
write_geotiff(os.path.join(OUT, "aspect.tif"), aspect, tf, meta)
write_geotiff(os.path.join(OUT, "plan_c.tif"), plan_c, tf, meta)
write_geotiff(os.path.join(OUT, "prof_c.tif"), prof_c, tf, meta)

np.save(os.path.join(OUT, "transform.npy"), np.array([tf.c, tf.f, tf.a, tf.e]))
print("Stage1 done. Stats: elev min/max =", np.nanmin(dem), np.nanmax(dem),
      "; slope mean =", round(np.nanmean(slope),2), "; TWI shape =", twi_low.shape)