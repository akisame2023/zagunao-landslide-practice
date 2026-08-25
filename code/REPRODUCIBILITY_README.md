# Reproducibility Package README — Zagunao Landslide Susceptibility Benchmark

Companion to: *Landslide Susceptibility Mapping Based on Random Forest and XGBoost in an Earthquake-Affected Alpine Basin* (research report v2; see `paper_draft_en.md`).

Zenodo archiving: **optional** (see §6; reserved DOI if published as a dataset).

---

## 1. Overview

This package contains the complete pipeline that produced every result reported in the research report and its supplementary material. All inputs are open-access; all random seeds are fixed (seed = 42); all stage outputs are persisted as CSV/JSON/GeoTIFF/PNG.

**Report numbers ↔ pipeline stages:**

| Report table | Produced by | Output file |
|------------------|-------------|-------------|
| Table 3 (random CV) | `stage13_final_v3.py` | `model_results_final_v3.csv` |
| Table 3b (significance/calibration) | `stage21b_significance_fixed.py` | `significance_tests_fixed.csv` |
| Table 3c (spatial CV schemes) | `stage18_rigorous_spatial_cv.py` | `cv_rigorous.json` |
| Table 4 (temporal extrapolation) | `stage19_temporal_extrapolation.py` | `temporal_extrapolation.csv` |
| Table 5 (NDVI ablation) | `stage20c_ndvi2007_process.py` + `stage23_ndvi_ablation.py` | `ndvi_ablation.json` |
| Table 6 (OOF zoning) | `stage22_oof_zoning.py` | `zoning_oof.csv`, `oof_zoning_summary.json` |
| Basin-mask zoning (Supplementary) | `stage24_basin_mask.py` | `zoning_basin_mask.csv`, `basin_mask.tif` |
| Figures 1–7 | `stage14_figs_v3.py`, `stage16_fig6.py`, `stage17_shap.py` | `fig*.png` |

## 2. Environment

- Python ≥ 3.10 (tested 3.13); packages: numpy, pandas, scikit-learn, xgboost, scipy, matplotlib, rasterio, geopandas, shap, pysheds, pyhdf (optional)
- HDF4 handling: QGIS 3.40 bundled GDAL (≥3.6 with HDF4 driver) is used by `stage10`/`stage20c` for MODIS HDF4 → GeoTIFF conversion (exact commands in scripts)
- OS: Windows 10/11 (paths are relative within package; see §5)

## 3. Data sources (all open-access)

| Dataset | URL | Stage |
|---------|-----|-------|
| Copernicus DEM GLO-30 (30 m) | https://copernicus-dem-30m.s3.amazonaws.com/ (AWS Open Data) | stage1 |
| MODIS MOD13Q1 NDVI (250 m) | https://lpdaac.usgs.gov/ (Earthdata account) | stage9/10 (2020), stage20 (2007) |
| Wenchuan landslide inventories (2008–2021) | Zenodo 10.5281/zenodo.16418155 (Chen Ming, 2025) | stage2, stage19 |
| GEM global active faults (used/diagnosed, excluded) | Zenodo 10.5281/zenodo.3376300 | stage11/12 diagnostics |

## 4. Pipeline (stages 1–24)

```
stage1_terrain.py          DEM mosaic, clip, terrain factors (elev/slope/aspect/curv/TWI/SPI/STI/relief)
stage2b_samples.py         Landslide-buffer negative sampling, 1:1 balance
stage5_full.py             Initial 11-factor modelling (exploratory)
stage8_final.py            v1 10-factor final (pre-NDVI)
stage9/10_ndvi*.py         MODIS download + HDF4 → 30 m NDVI (2020)
stage11_fault_ndvi.py      GEM fault-distance trial + diagnostics (excluded)
stage12_final_v2.py        12-factor trial incl. fault distance (documents pseudo-correlation)
stage13_final_v3.py        FINAL 11-class/12-feature model (Table 3)
stage14_figs_v3.py         Final zoning figures
stage15(+15b)_spatial_cv*  Spatial CV exploration (k-means & grid-snake) — historical schemes
stage16_fig6.py            CV-comparison figure
stage17_shap.py            SHAP explainability (Fig 7)
stage18_rigorous_spatial_cv.py  Rigorous spatial CV, 3 schemes + buffers + persisted folds (Table 3c)
stage19_temporal_extrapolation.py 2013–2021 temporal validation (Table 4)
stage20(c)_ndvi2007*.py    2007 pre-quake NDVI processing
stage21b_significance_fixed.py   McNemar / DeLong-style / Brier (Table 3b)
stage22_oof_zoning.py      Out-of-fold zoning (Table 6)
stage23_ndvi_ablation.py   2007-vs-2020 NDVI ablation (Table 5)
stage24_basin_mask.py      Basin delineation (pysheds) + mask zoning (Supplementary)
```

## 5. Paths & reproducibility notes

- All scripts read/write under a single `outputs/` root defined per script; edit the top `OUT =` constant if relocating.
- No absolute machine-specific paths remain (verified: no `C:\Users\` references in final stage scripts).
- Seeds: `SEED = 42` everywhere; StratifiedKFold shuffles fixed.
- HDF4 step (QGIS GDAL) is the only non-Python dependency; fallback: any GDAL build with HDF4 driver (see scripts for exact `gdal_translate`/`gdalwarp` calls).

## 6. Packaging for Zenodo

```
zip -r zagunao_landslide_benchmark.zip \
    paper_draft_en.md paper_draft_en_NH.md \
    experiment/stage*.py \
    experiment/outputs/*.csv *.json *.tif fig*.png
```

Upload to Zenodo with: title = report title; license = CC-BY-4.0 (data) / MIT (code); creators = [authors]. Reserve DOI if and when archiving; add DOI to the report's Data availability section.

## 7. Contact

[Author emails] — code issues via GitHub issues (repo to be created).