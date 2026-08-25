"""Stage 14: render final figures for v3 (11-factor incl. NDVI)."""
import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"

# fig5 zones v3
cmap = ListedColormap(["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"])
with rasterio.open(os.path.join(OUT, "zones_final_v3.tif")) as src:
    zones = src.read(1); tfz = src.transform
fig, ax = plt.subplots(figsize=(8.5, 7))
ax.imshow(zones, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest",
          extent=[tfz.c, tfz.c + tfz.a * zones.shape[1], tfz.f + tfz.e * zones.shape[0], tfz.f])
cbar = fig.colorbar(ax.images[0], ax=ax, ticks=range(5), shrink=0.75)
cbar.ax.set_yticklabels(["极低", "低", "中", "高", "极高"])
ax.set_xlabel("经度 / °E"); ax.set_ylabel("纬度 / °N")
ax.set_title("杂谷脑河流域滑坡易发性分区图（XGBoost，五级划分，11 因子）")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_zones_v3.png"), dpi=300); plt.close(fig)
print("fig5_zones_v3.png saved")

# check zone distribution sanity
with rasterio.open(os.path.join(OUT, "zones_final_v3.tif")) as src:
    z = src.read(1)
valid = z[z != 255]
for k in range(5):
    print(f"zone {k}: {(valid == k).sum() / valid.size * 100:.1f}%", end="  ")
print()
print("Stage14 done")