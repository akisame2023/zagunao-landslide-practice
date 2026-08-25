# 源数据清单（SOURCES）

> 本清单对应项目的全部原始输入数据。除滑坡编目 zip（已附于本目录）外，其余数据均可通过 `code/download_sources.py` 一键下载，或按表内 URL 手动获取。
> 体积合计约 2.2 GB；全部数据公开免费。

---

## 一、包内已附

| 数据 | 文件 | 体积 | 用途 |
|------|------|------|------|
| 滑坡编目 2008–2021 | `landslide_dataset.zip` | 50 MB | 训练（2008）与时间外推（2013–2021）；解压后含 `ls/` 下逐年 shapefile |

> 解压命令：`unzip landslide_dataset.zip`（Windows: 右键解压）；Shapefile 用 QGIS / geopandas 读取。

## 二、需下载（脚本或手动）

| 数据 | 来源 URL | 体积 | 用途 | 下载方式 |
|------|---------|------|------|---------|
| Copernicus DEM GLO-30（4 瓦片：N31E102/N31E103/N32E102/N32E103，30 m） | `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{TILE}_00_DEM/..._DEM.tif` | 157 MB | 全部地形因子（stage1） | `python download_sources.py --dem` 或 AWS 页面 |
| MODIS MOD13Q1 NDVI 2020 夏（5 期，h26v05） | `https://data.lpdaac.earthdatacloud.nasa.gov/...`（需 Earthdata 账号） | 1.16 GB | NDVI 因子（stage9/10） | `python download_sources.py --modis2020` |
| MODIS MOD13Q1 NDVI 2007 夏（3 期，h26v05） | 同上（2007 场景号见脚本） | 697 MB | NDVI 年份消融（stage20c/23） | `python download_sources.py --modis2007` |

## 三、Earthdata token（下载 MODIS 必需）

1. 注册/登录 [NASA Earthdata](https://urs.earthdata.nasa.gov/users/new)；
2. 在 [LP DAAC Token 页](https://lpdaac.usgs.gov/user/tokens) 生成 Token；
3. 运行下载前设置环境变量：`set EARTHDATA_TOKEN=你的token`（PowerShell: `$env:EARTHDATA_TOKEN="..."`）。

## 四、其余输入（源自公开数据，非原始下载）

| 内容 | 来源 |
|------|------|
| 岩性 / 断层（**诊断后弃用**） | GEM 全球活跃断层（Zenodo 10.5281/zenodo.3376300，2019.0）——仅用于伪相关诊断记录，非模型输入 |

## 五、数据校验建议

- 下载后核对文件体积（与表内一致、MODIS 约 230–250 MB/期）；
- 编目 zip 的 SHA256（可选）：`Get-FileHash landslide_dataset.zip -Algorithm SHA256`（发布时登记值，需要时可从 Zenodo 记录比对）。

## 六、引用建议

- DEM: Copernicus DEM (ESA/Copernicus Programme, AWS Open Data);
- NDVI: MODIS MOD13Q1 (NASA LP DAAC);
- 滑坡编目: Chen M. (2025), Zenodo DOI: 10.5281/zenodo.16418155.