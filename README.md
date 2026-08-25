# final/ — 科研练习项目交付包

> 项目：基于公开数据的 GIS+机器学习滑坡易发性研究流程示范（汶川震区杂谷脑河流域）
> 整理日期：2026-08-24 ｜ 用途：科研练习/课程作业/方法学训练
> 本目录包含一个完整研究流程的全部产物：研究报告（全文叙述）、图件、代码、关键数据与审查记录。

---

## 快速入口

| 想做什么 | 看哪个 |
|---------|--------|
| 了解项目定位与学习路径 | **`科研练习项目说明书.md`** |
| 从零复现全流程 | **`复现教程.md`** |
| 阅读完整研究叙述 | `paper_draft_en.md`（项目报告全文） |
| 获取源数据 | **`data/SOURCES.md`** + `code/download_sources.py`（一键下载） |
| 跑代码 | `code/REPRODUCIBILITY_README.md`（环境+stage↔结果映射） |
| 核对数据 | `data/` + `修订执行记录.md` + `图件审查报告.md` |

---

## 一、研究报告（根目录）

| 文件 | 说明 |
|------|------|
| `paper_draft_en.md` | 完整研究叙述：背景、数据、方法、结果、讨论、结论（含全部表 1/2/3/3b/3c/4/5/6 与 8 张图引用） |

## 二、图件（figures/，9 张）

| 图 | 内容 |
|----|------|
| fig1_flow.png | 技术路线 7 步 |
| fig2_study_area.png | 研究区 + 2008 滑坡分布（13,288 编目 / 13,086 建模注） |
| fig3_roc_v3.png | ROC（0.873/0.902/0.910） |
| fig4_importance_v3.png | 特征重要性（Gini/gain） |
| fig5_zones_v3.png | 五级分区图 |
| fig6_cv_comparison.png | 随机 vs 空间 CV（0.873/0.902/0.910 vs 0.827/0.819/0.819） |
| fig7_shap_summary/bar/dependence.png | SHAP 三图 |

## 三、代码（code/，17 个 stage + README）

完整可复现流水线（stage 1 → 26）。注意：代码内 `OUT =` 路径为原机器绝对路径，重新运行前请改为本地路径（README §5 说明）。

## 四、关键数据（data/，8 个文件）

- `model_results_final_v3.csv` — 随机 5 折 CV（XGB 0.910±0.006）
- `cv_rigorous.json` — 三方案空间 CV + 每折组成
- `temporal_extrapolation.csv` — 2013–2021 时间外推
- `significance_tests_fixed.csv` — McNemar/DeLong/Brier
- `ndvi_ablation.json` — 2007 vs 2020 NDVI 消融
- `oof_zoning_summary.json` + `zoning_oof.csv` — OOF 分区
- `zoning_basin_mask.csv` — 流域掩膜分区

## 五、审查记录（根目录）

- `修订执行记录.md` — 方法修订与自查逐项记录
- `图件审查报告.md` — 图件视觉 + 代码审查记录