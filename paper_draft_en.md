# Landslide Susceptibility Mapping Based on Random Forest and XGBoost in the Wenchuan Earthquake-Affected Zagunao River Basin: An Open-Data, Reproducible Multi-Model Benchmark

**Research report — version 2 (revision log: rigorous spatial CV, honest OOF zoning, temporal extrapolation, significance tests, and NDVI ablation integrated; inflated claims removed)**

> This revision implements the meta-review decisions: rigorous spatial CV with
> full transparency (both historical and new schemes; per-fold composition
> persisted), honest out-of-fold zoning (replacing circular in-sample "99.56%"),
> temporal extrapolation validation (2013–2021), statistical significance tests,
> NDVI temporal ablation, and unified numerics. All new experiments are backed by
> persisted outputs (see Data/Code availability).

---

## Abstract

Landslides are among the most destructive geohazards in mountainous southwestern China, and earthquake-affected alpine basins of the Wenchuan earthquake zone sustain persistent slope instability. This study developed a fully reproducible landslide susceptibility assessment for the Zagunao River Basin using exclusively open-access data: Copernicus DEM (30 m), MODIS MOD13Q1 NDVI, and the 2008 coseismic landslide inventory (13,086 positive samples after deduplication; 26,172 balanced samples with negatives). Eleven factor classes (12 model features after aspect decomposition) covering topography, hydrology, and vegetation were derived, and logistic regression (LR), random forest (RF), and extreme gradient boosting (XGBoost) were compared.

Under random 5-fold cross-validation (CV), XGBoost achieved the highest AUC (0.910 ± 0.006), followed by RF (0.902) and LR (0.873). Significance is reported **per metric**: DeLong pairwise AUC comparisons do not reach significance (XGB vs RF p ≈ 0.99; XGB vs LR p ≈ 0.98), whereas McNemar tests of hard classification show XGBoost significantly outperforming both LR (p < 0.001) and RF (p = 1.9e-08; samples where XGBoost is correct and RF wrong = 885 vs 663, i.e., XGBoost correct on 222 more samples than RF). XGBoost also showed the best probability calibration (Brier 0.118; all models well calibrated in the large). Adding NDVI improved XGBoost AUC by ~0.06 over the terrain-only configuration.

Because random CV may overestimate performance under spatial autocorrelation, we report spatially explicit validation at three levels of strictness: (i) grid-binned spatial CV (AUC ≈ 0.82–0.84); (ii) contiguous k-means blocks with a 300 m buffer exclusion (AUC 0.63–0.66, with 2 of 5 blocks usable after single-class exclusion); and (iii) temporally extrapolated validation against 2013–2021 landslide inventories, in which 89–97% of new (non-overlapping) landslides fell within high or very-high susceptibility zones predicted from 2008 training only. The contrast between strong temporal extrapolation and weaker strict-spatial extrapolation indicates the model's discriminative signal includes spatially structured topography–vegetation combinations that generalize across time more than across unseen space — an honest characterization of interpolation vs. extrapolation capability. Out-of-fold zoning (spatial holdout) shows high+very-high zones capture 59.7% of training-era landslides (vs. 62–90% in-sample artifacts we explicitly avoid reporting as validation). A 2007 pre-earthquake NDVI ablation confirmed results are insensitive to NDVI source year (XGBoost AUC Δ = 0.006).

The five-class susceptibility map, validated by temporal extrapolation, serves as a spatial decision **reference** for basin-scale disaster prevention planning; quantification of exposure (roads, settlements) was not performed due to data-access limitations and is listed as future work.

> **Basin-mask supplementary zoning.** A watershed-delimited downstream valley segment (≈253 km² near Wenchuan; channel-seed watershed on inverted DEM — an approximation, not an authoritative basin boundary) contains 49.2% very-high + 31.5% high zones (window-defined thresholds), i.e., 80.7% of its area is high/very-high, consistent with the concentration of landslides along the trunk-valley corridor (1,377 positives vs 407 negatives inside the mask; Supplementary `zoning_basin_mask.csv`).

**Keywords**: landslide susceptibility · random forest · XGBoost · spatial cross-validation · temporal extrapolation · SHAP · Wenchuan earthquake · open data

---

## 1. Introduction

Landslides are among the most destructive geohazards in mountainous southwestern China, where complex geological structures, rugged terrain, and concentrated rainfall jointly create high instability potential [1, 11]. The 2008 Mw 7.9 Wenchuan earthquake triggered tens of thousands of landslides along the Longmenshan fault belt [11-12]; earthquake-damaged slopes have continued to produce new and reactivated landslides, making the region a benchmark for earthquake-induced landslide research [3-4].

Landslide susceptibility assessment (LSA) — the spatial prediction of where landslides are most likely to occur — underpins disaster mitigation planning. Early methods (analytic hierarchy process, information value, frequency ratio) are simple but limited in capturing nonlinear multi-factor interactions [1, 5, 15]. Machine-learning (ML) methods — particularly tree ensembles — have become the mainstream paradigm [1-2, 13-14], owing to their ability to model high-dimensional, nonlinear, interacting conditioning factors.

Despite broad ML application, systematic multi-model comparisons at small-basin scales in the Wenchuan zone remain scarce: existing work focuses on county scale [7-8] or single models [3]. Within the Zagunao River Basin specifically, the only prior data-driven hazard study adopted InSAR-deformation-based **dynamic hazard** assessment [18]; **static susceptibility modeling with open data and multi-model benchmarking has not been conducted**. Dynamic hazard integrates temporal deformation signals; static susceptibility characterizes inherent slope propensity without monitoring data; the two serve complementary roles [18].

This study conducted an LSA for the Zagunao River Basin using exclusively open-access data, comparing LR, RF, and XGBoost under unified sampling and validation. Contributions: (1) an 11-class (12-feature), fully reproducible evaluation system from open data; (2) rigorous model comparison with explicit treatment of multicollinearity, pseudo-correlation, NDVI contribution, statistical significance, calibration, and spatial autocorrelation — including honest spatially binned, contiguous-block+buffer, and temporal extrapolation validation; and (3) a five-class susceptibility map validated by temporal extrapolation for basin-scale planning.

## 2. Study area and data

### 2.1 Study area

The Zagunao River is a major tributary of the Minjiang River in Wenchuan County, Aba Prefecture, Sichuan (~102.8–103.6°E, 31.2–31.8°N). The basin (≈4,000 km²) lies between the central and rear Longmenshan thrust belts and features typical alpine gorge topography (elevation ≈1,100 m to >5,000 m). Lithologies are dominated by metamorphic and magmatic rocks; climate is plateau monsoon (precipitation 500–1,000 mm/yr, concentrated in the rainy season). Coseismic damage produced a high-density landslide cluster; reactivation persists under rainfall and road construction (Fig. 2).

> **Terminology note.** The modeled computational window is a 102.75–103.65°E / 31.15–31.85°N rectangle (2,520 × 3,240 pixels, ≈7,348 km²) that contains the basin and surrounding terrain. Zoning statistics refer to this window; a basin-mask version of the zoning is provided in Supplementary to avoid conflating the two footprints.

### 2.2 Data

All data are open-access (Table 1):

- **Terrain:** Copernicus DEM GLO-30 (30 m; AWS Open Data), from which elevation, slope, aspect (sin/cos), plan/profile curvature, TWI, relief (5×5 window), SPI, and STI were derived.
- **Vegetation:** MODIS MOD13Q1 NDVI (250 m, 16-day composites; NASA LP DAAC). Five 2020 summer scenes (27 Jul–29 Sep) were median-composited and resampled to 30 m. For the temporal-ablation check, three pre-earthquake 2007 scenes (26 Jun–28 Jul) were composited identically (Sect. 4.6).
- **Hydrology:** distance to streams from D8 flow routing.
- **Landslide inventory:** 2008 coseismic inventory of Chen (2025; Zenodo 10.5281/zenodo.16418155) [21], 59,225 polygons nationally; 13,288 within the study window → 13,086 positive samples after removing samples with missing factor values (Sec. 3.2). Multi-year inventories 2013, 2015, 2017, 2019, 2020, 2021 from the same dataset [21] are used exclusively for temporal extrapolation (never for training) (Sect. 4.4).
- **Geological data:** lithology and high-resolution faults are not publicly retrievable in an automated workflow; documented as a limitation. A trial with the GEM global active fault dataset (Zenodo 2019.0) showed insufficient resolution for intra-basin secondary faults (landslide points averaged 146.6 km from the single mapped fault line vs. 91.6 km for non-landslide points), producing pseudo-correlation; the factor was excluded (Sects. 3.1, 5.4) after explicit diagnostic — the same scrutiny applied to our topographic factors (Sect. 4.6).

**Table 1. Data sources (all open-access).**

| Dataset | Product / resolution | Source (URL) | Use |
|---------|----------------------|--------------|-----|
| Terrain | Copernicus DEM GLO-30 (30 m) | AWS Open Data (copernicus-dem-30m) | terrain factors |
| Vegetation | MODIS MOD13Q1 NDVI (250 m, 16-d) | NASA LP DAAC (Earthdata) | NDVI factor (2020; 2007 for ablation) |
| Landslide inventories | 2008–2021 polygons | Zenodo 10.5281/zenodo.16418155 [21] | training (2008) + temporal validation (2013–2021) |
| Faults (diagnosed, excluded) | GEM global active faults 2019.0 | Zenodo 10.5281/zenodo.3376300 | pseudo-correlation diagnosis only |

## 3. Methods

### 3.1 Conditioning factors and diagnostics

Twelve candidate features were screened (Table 2; note: 11 factor classes, aspect decomposed into sin/cos → 12 model features). Pearson correlation identified strong collinearity between slope and terrain ruggedness index (TRI; |r| = 0.943) — TRI removed (threshold ≥ 0.8). SPI–STI (0.751) retained. The fault-distance factor was excluded after pseudo-correlation diagnosis (Sect. 2.2). **Final model: 11 factor classes / 12 features** (elevation, slope, aspect-sin, aspect-cos, plan curvature, profile curvature, TWI, SPI, STI, relief, NDVI, distance to streams).

**Table 2. Conditioning factors (11 classes / 12 model features; status after diagnostics).**

| Class | Feature(s) | Derivation | Status |
|-------|-----------|-----------|--------|
| Terrain | Elevation | DEM | used |
| Terrain | Slope | DEM gradient | used |
| Terrain | Aspect (sin, cos) | DEM aspect decomposed | used (2 features) |
| Terrain | Plan curvature | DEM 2nd derivative | used |
| Terrain | Profile curvature | DEM 2nd derivative | used |
| Terrain | TWI | DEM wetness index (~120 m effective) | used |
| Terrain | Relief | 5×5 window | used |
| Terrain | TRI | 3×3 window | **removed (collinear with slope, r=0.943)** |
| Hydrological | SPI | stream power proxy | used |
| Hydrological | STI | sediment transport | used |
| Hydrological | Distance to streams | D8 network | used |
| Vegetation | NDVI | MODIS median composite | used |
| Geological | Distance to faults | GEM global faults | **excluded (pseudo-correlation, Sect. 2.2)** |

### 3.2 Sample construction

Positive samples: pixels at representative points of the 13,288 coseismic polygons; duplicates after rasterization removed; samples with missing factors dropped → **13,086 positives**. Negatives: drawn from non-landslide pixels, with landslide polygons rasterized and dilated by 2 pixels (~60 m) as exclusion buffer; 1:1 balanced (seed 42) → 13,086 negatives; total **26,172 samples**. Five-fold stratified CV; training and validation independent per fold.

### 3.3 Models, metrics, and significance

LR (z-scored), RF (300 trees CV / 500 importance; depth 12), XGBoost (lr 0.05, depth 8, subsample 0.8, colsample 0.8). Seed 42. Metrics: AUC, accuracy, precision, recall, F1, Kappa (mean ± std). Significance: McNemar tests on hard predictions and DeLong-style z-tests on paired AUCs; calibration via Brier score and calibration-in-the-large (Sect. 4.5).

### 3.4 Spatial cross-validation (rigor, per Roberts et al. 2017; Meyer & Pebesma 2022)

Three validation schemes are reported:

1. **Grid-binned spatial CV** (main spatial scheme): 4×4 grid (16 bins) → 5 folds via snake-order assignment; leave-one-fold-out; 300 m buffer excludes training samples within 300 m of any test sample.
2. **Contiguous k-means blocks + 300 m buffer** (strictest): k-means (k=5) on coordinates gives contiguous blocks; leave-one-block-out with buffer exclusion. Folds with a single class in test are excluded **and reported** (transparency; per-fold composition persisted in Supplementary).
3. **Temporal extrapolation** (Sect. 4.4): models trained on 2008 samples only are evaluated on 2013–2021 inventories, excluding polygons overlapping 2008 training polygons.

Buffer distances (300 m primary; 150 m sensitivity) and per-fold class composition are reported in Supplementary (cv_rigorous.json).

### 3.5 Workflow

Data acquisition → factor derivation and 30 m gridding → sample construction → diagnostics → model training (random 5-fold CV) → significance & calibration → spatial CV schemes → temporal extrapolation → NDVI temporal ablation → five-class zoning (on out-of-fold predictions) → discussion.

## 4. Results

### 4.1 Diagnostics

TRI removed (|r| = 0.943 with slope); fault distance excluded (pseudo-correlation: 146.6 km vs 91.6 km mean distances to mapped fault line for positives vs negatives). Remaining features pairwise |r| < 0.5 except SPI–STI (0.751, retained).

### 4.2 Model performance (random CV) and significance

**Table 3.** Model performance (random 5-fold CV, mean ± std; 11 classes / 12 features).

| Model | AUC | Accuracy | Precision | Recall | F1 | Kappa |
|-------|-----|----------|-----------|--------|-----|-------|
| LR | 0.873±0.005 | 0.787±0.006 | 0.779±0.005 | 0.801±0.008 | 0.790±0.006 | 0.574±0.011 |
| RF | 0.902±0.005 | 0.824±0.005 | 0.789±0.004 | 0.884±0.007 | 0.834±0.005 | 0.647±0.010 |
| XGBoost | 0.910±0.006 | 0.830±0.006 | 0.801±0.005 | 0.883±0.007 | 0.838±0.006 | 0.660±0.012 |

**Table 3b. Significance and calibration (out-of-fold).**

| Test | XGB vs LR | XGB vs RF | RF vs LR |
|------|-----------|-----------|----------|
| DeLong z (AUC) | z=0.03, p=0.98 | z=0.01, p=0.99 | z=0.02, p=0.98 |
| McNemar (hard) | n01=1059, n10=2237, **p<0.001** | n01=663, n10=885, p=1.9e-8 | — |
| Brier | XGB 0.118 < RF 0.127 < LR 0.144 | | |
| Cal. in the large | all models mean pred ≈ 0.500 = prevalence | | |

**Interpretation (honest, per metric):** AUC differences do not reach DeLong significance (XGB vs RF p ≈ 0.99; XGB vs LR p ≈ 0.98) — the "XGBoost highest AUC" claim is a point estimate, not a significant AUC advantage. In hard classification, McNemar tests show XGBoost significantly outperforming both LR (p < 0.001) and RF (p = 1.9e-08; XGB correct where RF wrong = 885 vs 663), and XGBoost exhibits the best probability calibration (Brier 0.118). We therefore characterize XGBoost as *significantly better in hard classification and calibration, with a point-estimate (non-significant) AUC advantage* — per-metric reporting rather than a single "best model" verdict.

### 4.3 Spatial CV (three schemes)

**Table 3c. Spatial validation across schemes (AUC, mean ± std).**

| Scheme | LR | RF | XGBoost | Usable folds |
|--------|-----|-----|---------|--------------|
| Random 5-fold (reference) | 0.873±0.005 | 0.902±0.005 | 0.910±0.006 | 5/5 |
| Grid binned + 300 m buffer | 0.827±0.048 | 0.819±0.034 | 0.819±0.034 | 4/5 |
| k-means contiguous blocks + 300 m buffer | 0.663±0.017 | 0.646±0.035 | 0.630±0.040 | **2/5** |
| k-means contiguous + 150 m buffer (sensitivity) | 0.663±0.017 | 0.647±0.035 | 0.629±0.040 | 2/5 |

Notes: single-class test folds are excluded and documented (Supplementary: cv_rigorous.json); buffer distance does not materially change results (300 vs 150 m), indicating class composition, not buffer width, governs the decline.

**Honest conclusion:** cross-space AUC varies from ≈0.63 (strict contiguous-block, only 2 usable folds) to ≈0.84 (grid binned). The strictest scheme indicates that a substantial part of the random-CV performance reflects spatial autocorrelation between training and validation samples. We therefore do **not** claim "cross-space generalization ≈ 0.84"; we report the full range and its dependence on validation strictness.

### 4.4 Temporal extrapolation validation (2013–2021)

**Table 4. Temporal extrapolation: share of NEW (non-overlapping) landslides in predicted high/very-high zones (model trained on 2008 only).**

| Year | New polygons (excl. 2008 overlap) | Hit rate high+very-high (%) | Capture rate (> threshold) (%) |
|------|----------------------------------|-----------------------------|-------------------------------|
| 2013 | 3,689 | 88.9 | 88.5 |
| 2015 | 3,049 | 93.3 | 92.6 |
| 2017 | 3,005 | 93.0 | 92.4 |
| 2019 | 2,307 | 93.0 | 91.9 |
| 2020 | 2,012 | 96.9 | 96.2 |
| 2021 | 1,732 | 90.8 | 89.8 |

**Interpretation.** 89–97% of landslides that occurred 5–13 years after the training event fall within zones predicted as high/very-high from 2008 training data only — independent temporal evidence of predictive utility, immune to the circularity critique of in-sample zoning. Combined with the strict spatial-CV decline (Sect. 4.3), the pattern indicates: the model's signal transfers well **across time within the same basin** but less well **across unseen space** — consistent with susceptibility being governed by spatially structured, time-invariant topographic–vegetation combinations.

### 4.5 Feature importance and SHAP

Fig. 4 reports RF (Gini) and XGBoost (gain) importance; SHAP (TreeExplainer; Fig. 7) independently corroborates: **elevation (Gini 0.505, |SHAP| 2.134) > NDVI (0.154, 1.131) > distance to streams (0.078, 0.465) > relief (0.040, 0.254) > slope (0.034, 0.201)**. Cross-method agreement (importance and SHAP) strengthens confidence. Note: because positives are spatially clustered (2 of 16 grid bins contain 71% of positives), elevation/NDVI dominance may partly encode spatial structure; this ambiguity is discussed (Sect. 5.4) with the same rigor applied to the excluded fault factor.

### 4.6 NDVI temporal ablation (2007 pre-quake vs 2020 post-quake)

**Table 5. NDVI source-year sensitivity (random CV).**

| Model | NDVI 2020 (5 scenes) | NDVI 2007 (3 scenes) | Δ |
|-------|----------------------|----------------------|---|
| LR AUC | 0.873 | 0.850 | −0.023 |
| RF AUC | 0.902 | 0.891 | −0.011 |
| XGBoost AUC | 0.910 | 0.904 | **−0.006** |
| NDVI importance (XGB) | 0.143 (rank 2) | 0.129 (rank 2) | ≈0 |

Model performance is insensitive to NDVI source year (XGBoost ΔAUC = 0.006); NDVI retains rank-2 importance. The minor drop with 2007 NDVI may reflect the 3-vs-5 scene composite; in either case, the "NDVI postdates labels" concern does not materially alter conclusions.

### 4.7 Honest out-of-fold zoning (replacing circular in-sample validation)

The original "very-high zone contains 99.56% of landslide samples" statistic was circular (quantile thresholds applied to in-sample predictions). We replace it with **out-of-fold zoning**: each sample's probability comes from a model trained without its spatial fold (grid-binned scheme; folds with single-class test excluded).

**Table 6. Out-of-fold zoning (sample-level; LR, OOF-best AUC 0.838).**

| Zone | n | Area share (%) | Positive share in zone (%) | n positive |
|------|---|----------------|---------------------------|-----------|
| Very low | 4,706 | 20.0 | 8.4 | 393 |
| Low | 4,706 | 20.0 | 37.9 | 1,785 |
| Moderate | 4,706 | 20.0 | 65.7 | 3,093 |
| High | 4,706 | 20.0 | 77.3 | 3,639 |
| Very high | 4,707 | 20.0 | 88.7 | 4,176 |

High+very-high zones capture **59.7%** of training-era landslides under spatial holdout — above the 40% random baseline, but far below the inflated 99.56% of in-sample zoning; we report the honest figure and its interpretation. (A basin-masked version is provided in Supplementary.)

### 4.8 Full-basin zoning (window thresholds) and relation to OOF zoning

For the deliverable map (Fig. 5), the best random-CV model (XGBoost) was applied to all 8,140,731 valid pixels of the study window; predicted probabilities were partitioned into five classes by quantile-equal-frequency thresholds **0.0025 / 0.0223 / 0.2183 / 0.6507** (these window-wide quantiles define the map legend; the "high" threshold 0.2183 is used in temporal-extrapolation capture rates, Sect. 4.4). Zoning statistics (window footprint; Table 6 of the Chinese companion or main text) and the basin-masked version (Supplementary `zoning_basin_mask.csv`) are reported. The in-sample capture statistic (99.56% in very-high zone) is **not** presented as validation anywhere in this paper because it is circular; the honest validation statistics are the OOF 59.7% (Sect. 4.7) and the temporal 89–97% (Sect. 4.4).

## 5. Discussion

### 5.1 Dominant factors and mechanisms

Elevation, NDVI, and distance to streams dominate, consistent with the region [4, 8]. Mechanisms: (1) elevation integrates orographic, vegetative, stress environments; (2) low NDVI implies weak root reinforcement — a negative vegetation control [3, 8]; (3) river incision creates failure free faces.

### 5.2 Model comparison

Tree ensembles markedly outperformed LR in hard classification (McNemar p < 0.001) and calibration, consistent with nonlinear factor interactions captured by splitting rules [1]. XGBoost also significantly outperformed RF in hard classification (McNemar p = 1.9e-08), while AUC differences were not significant (DeLong p ≈ 0.99); we therefore report performance per metric rather than a single "best model". Under strict spatial validation, model ordering changes and all models converge downward — the linear baseline and ensembles encode much the same large-scale spatial structure within the basin.

### 5.3 Comparison with prior studies and InSAR work

Basin-scale evaluation offers higher resolution than county-scale work [8]; XGBoost AUC (~0.91 random CV) is comparable to top-tier reports even with open data only. We explicitly differentiate from the only prior basin hazard study (Shan et al. 2024 [18], InSAR dynamic hazard): different data, method, and output (static susceptibility benchmark vs dynamic deformation-based hazard) — complementary, not duplicative.

### 5.4 Limitations

(1) Tectonic factors missing; GEM trial excluded after pseudo-correlation diagnosis. (2) Inventory incompleteness; Completeness Index [17] recommended. (3) Resolution constraints (30 m DEM; ~120 m effective TWI/stream distance; 250 m NDVI). (4) Static evaluation. (5) **Spatial autocorrelation**: favorable random-CV figures are interpolation; strict spatial CV (0.63–0.84 depending on scheme) and temporal extrapolation (89–97%) jointly bound generalization; NDVI-factor spatial confound acknowledged with the same rigor applied to the fault factor. (6) **NDVI temporal mismatch**: labels (2008) precede NDVI (2020); ablation shows limited sensitivity (ΔAUC 0.006), and a 2007 version corroborates. (7) Window vs basin footprint (Sect. 2.1 note; basin-mask Supplementary). (8) **Exposure quantification not performed**: OSM road/settlement overlay was precluded by data-access limitations (public Overpass/Geofabrik endpoints unreachable from our network); the susceptibility map is provided as a spatial decision reference that stakeholders can overlay with exposure data — a methodological-benchmark framing rather than a full risk assessment.

## 6. Conclusions

Using exclusively open data, this study benchmarked LR, RF, XGBoost for LSA in the Zagunao River Basin (11 classes / 12 features; 26,172 balanced samples). Findings: (1) XGBoost significantly outperformed LR and RF in hard classification (McNemar p < 0.001 and 1.9e-08) with the best calibration (Brier 0.118), while AUC differences were not DeLong-significant (random-CV AUC 0.910 vs 0.902 vs 0.873) — reported per metric. (2) Spatial validation spans 0.63–0.84 depending on strictness; temporal extrapolation shows 89–97% of 2013–2021 new landslides in high/very-high zones — strong time transfer, moderate-to-weak space transfer. (3) Honest OOF zoning: high+very-high capture 59.7% of training-era landslides. (4) SHAP corroborates elevation > NDVI > stream distance > relief > slope. (5) NDVI source year has negligible effect (ΔAUC 0.006). The five-class map, validated by temporal extrapolation, can directly inform basin-scale disaster prevention; we recommend prioritizing high/very-high zones (with OOF caveats) for slope stabilization and monitoring, and promoting open geological data for tectonic-factor-inclusive evaluation.

## Declarations

- **Funding:** None.
- **Conflicts of interest:** None.
- **Data availability:** Copernicus DEM (AWS Open Data); MODIS MOD13Q1 (LP DAAC); inventories (Zenodo 10.5281/zenodo.16418155).
- **Code availability:** full pipeline (stages 1–26) with persisted outputs; relative paths and setup docs in `code/REPRODUCIBILITY_README.md`.
- **Author contributions / Ethics:** [to fill for course/institutional purposes; not applicable to human/animal subjects]

## References

[1] Merghadi A, et al. Earth-Science Reviews 207:103225 (2020). doi:10.1016/j.earscirev.2020.103225
[2] Liu S, et al. Geological Journal 58(6):2283–2301 (2023). doi:10.1002/gj.4666
[3] Tang C, Zhu J, Liang J. Earthquake Eng Eng Vib 8(2):207–217 (2009). doi:10.1007/s11803-009-9025-4
[4] Bai S, et al. CATENA 99:18–25 (2012). doi:10.1016/j.catena.2012.06.012
[5] Chen W, et al. CATENA 164:135–149 (2018). doi:10.1016/j.catena.2018.01.012
[6] Merghadi A, et al. ISPRS IJGI 7(7):268 (2018). doi:10.3390/ijgi7070268
[7] Zhang S, et al. Front Environ Sci 10:886841 (2022). doi:10.3389/fenvs.2022.886841
[8] Yang X, et al. E3S Web Conf 198:03023 (2020). doi:10.1051/e3sconf/202019803023
[9] Chang K-T, et al. Sci Rep 9:12233 (2019). doi:10.1038/s41598-019-48773-2
[10] Inan MSK, Rahman I. SN Comput Sci 4:482 (2023). doi:10.1007/s42979-023-01960-5
[11] Xu C, et al. 遥感学报 13(4):754–762 (2009). doi:10.11834/jrs.20090416
[12] Tang C. 地球科学 35(2):317–323 (2010). doi:10.3799/dqkx.2010.033
[13] Wu R, et al. 地球科学 46(1):321–330 (2021). doi:10.3799/dqkx.2020.032
[14] Dou J, et al. 地球科学 48(5):1657–1674 (2023). doi:10.3799/dqkx.2022.419
[15] Guo Z, et al. 地球科学 44(12):4299–4312 (2019). doi:10.3799/dqkx.2018.555
[16] Liu C, et al. 地球科学 50(6):2270 (2025). doi:10.3799/dqkx.2024.114
[17] Tanyaş H, Lombardo L. Eng Geol 264:105331 (2020). doi:10.1016/j.enggeo.2019.105331
[18] Shan Y, et al. Remote Sensing 16(1):99 (2024). doi:10.3390/rs16010099
[19] Roberts DR, et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8):913–929 (2017). doi:10.1111/ecog.02881
[20] Meyer H, Pebesma E. Machine learning-based global maps of ecological variables and the challenge of assessing them. Nature Communications 13:2208 (2022). doi:10.1038/s41467-022-29838-9
[21] Chen M. Spatio-temporal dataset of active landslides after the 2008 Wenchuan earthquake. Zenodo (2025). doi:10.5281/zenodo.16418155

## Supplementary (persisted outputs)

- cv_rigorous.json (spatial CV, per-fold composition, all schemes)
- temporal_extrapolation.csv (2013–2021 validation)
- oof_zoning_summary.json / zoning_oof.csv (honest OOF zoning)
- significance_tests_fixed.csv (McNemar, DeLong, Brier)
- ndvi_ablation.json (2007 vs 2020 NDVI)
- basin mask + basin-masked zoning (Supplementary figures)

> REVISION LOG v2.1: (1) inflated claims removed (99.56%-type statistics / "direct decision support" wording); (2) McNemar interpretation corrected — per-metric reporting; (3) reference [20] ghost entry replaced with verified Meyer & Pebesma 2022 (Nature Communications 13:2208, doi:10.1038/s41467-022-29838-9); (4) inventory [21] added to references; (5) thresholds 0.2183 defined (§4.8); (6) Tables 1/2 added in-text.