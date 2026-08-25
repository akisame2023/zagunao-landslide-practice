"""
download_sources.py — 一键获取本项目全部源数据（公开来源）

用法:
    python download_sources.py --dem              # Copernicus DEM 4 瓦片 (~157 MB)
    python download_sources.py --modis2020        # 2020 夏 5 期 NDVI (~1.16 GB, 需 Earthdata token)
    python download_sources.py --modis2007        # 2007 震前 3 期 NDVI (~697 MB, 需 token)
    python download_sources.py --inventory        # 滑坡编目 zip (~50 MB, 包内 data/ 已有则跳过)
    python download_sources.py --all              # 全部 (需设置 EARTHDATA_TOKEN)

Earthdata token 获取: 登录 https://lpdaac.usgs.gov/ → Generate Token
已下载文件会跳过（断点续传式检查），重新下载请先删旧文件。

输出目录:
    --out DEM_DIR 数据根目录，默认 ./sources （相对本脚本所在目录）
"""
import os
import sys
import argparse
import time
import urllib.request

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources")
TOKEN = os.environ.get("EARTHDATA_TOKEN", "").strip()

DEM_TILES = ["N31E102", "N31E103", "N32E102", "N32E103"]
DEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

MOD13Q1_BASE = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13Q1.061"
MODIS_2020 = [
    "MOD13Q1.A2020209.h26v05.061.2020342021543",
    "MOD13Q1.A2020225.h26v05.061.2020344204509",
    "MOD13Q1.A2020241.h26v05.061.2020346093237",
    "MOD13Q1.A2020257.h26v05.061.2020347062509",
    "MOD13Q1.A2020273.h26v05.061.2020349200629",
]
MODIS_2007 = [
    "MOD13Q1.A2007177.h26v05.061.2021068095141",
    "MOD13Q1.A2007193.h26v05.061.2021068163334",
    "MOD13Q1.A2007241.h26v05.061.2021073200334",
]
INVENTORY_URL = ("https://zenodo.org/records/16418155/files/"
                 "landslide%20dataset.zip/content")


def fetch(url, dest, token=None, min_size=0, tries=4, timeout=1800):
    """下载文件，支持重试与最小体积校验。"""
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"  [skip] 已存在: {os.path.basename(dest)}")
        return True
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) geo-research/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
            t0 = time.time()
            with opener.open(req, timeout=timeout) as r, open(dest, "wb") as f:
                total = 0
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    if total % (32 << 20) < (1 << 16):
                        print(f"    ...{total/1e6:.0f} MB "
                              f"({total/1e6/(time.time()-t0):.1f} MB/s)")
            if min_size and total < min_size:
                print(f"  [short] 仅 {total/1e6:.0f} MB (<{min_size/1e6:.0f} MB)，重试")
                os.remove(dest)
                continue
            print(f"  [OK] {os.path.basename(dest)} {total/1e6:.0f} MB")
            return True
        except Exception as e:
            print(f"  [try{i+1}] 失败: {type(e).__name__} {str(e)[:70]}")
            time.sleep(6)
    print(f"  [FAIL] {os.path.basename(dest)}")
    return False


def download_dem(out):
    print("== Copernicus DEM (30m) ==")
    d = os.path.join(out, "dem")
    os.makedirs(d, exist_ok=True)
    ok = True
    for t in DEM_TILES:
        name = f"Copernicus_DSM_COG_10_{t}_00_DEM"
        url = f"{DEM_BASE}/{name}/{name}.tif"
        ok &= fetch(url, os.path.join(d, f"{t}.tif"), min_size=30_000_000)
    return ok


def download_modis(out, scenes, label):
    print(f"== MODIS {label} NDVI ==")
    if not TOKEN:
        print("  !! 未设置 EARTHDATA_TOKEN，跳过（参考文件头说明）")
        return False
    d = os.path.join(out, f"modis{label}")
    os.makedirs(d, exist_ok=True)
    ok = True
    for sid in scenes:
        url = f"{MOD13Q1_BASE}/{sid}/{sid}.hdf"
        ok &= fetch(url, os.path.join(d, f"{sid}.hdf"), token=TOKEN,
                    min_size=200_000_000)
    return ok


def download_inventory(out):
    print("== 滑坡编目 (Zenodo) ==")
    d = os.path.join(out, "inventory")
    os.makedirs(d, exist_ok=True)
    return fetch(INVENTORY_URL, os.path.join(d, "landslide_dataset.zip"),
                 min_size=45_000_000)


def main():
    ap = argparse.ArgumentParser(description="获取项目全部公开源数据")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dem", action="store_true")
    ap.add_argument("--modis2020", action="store_true")
    ap.add_argument("--modis2007", action="store_true")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    if not any([args.all, args.dem, args.modis2020, args.modis2007, args.inventory]):
        ap.print_help()
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    if args.all or args.dem:
        download_dem(args.out)
    if args.all or args.modis2020:
        download_modis(args.out, MODIS_2020, "2020")
    if args.all or args.modis2007:
        download_modis(args.out, MODIS_2007, "2007")
    if args.all or args.inventory:
        download_inventory(args.out)
    print("\n完成。目录结构：")
    for root, _, files in os.walk(args.out):
        n = sum(1 for _ in files)
        if n:
            print(f"  {os.path.relpath(root, args.out)}/ ({n} 文件)")


if __name__ == "__main__":
    main()