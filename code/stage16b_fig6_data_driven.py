"""
Stage 16b (v2.1): render spatial-CV vs random-CV comparison bar chart (Fig 6).
Reads numbers from persisted outputs (no hardcoding):
- Random 5-fold: model_results_final_v3.csv
- Spatial (grid binned, 300m buffer): cv_rigorous.json scheme_B_grid_snake_300m
This guarantees figure-manuscript consistency.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False
OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"

# random CV from final model CSV
res = pd.read_csv(os.path.join(OUT, "model_results_final_v3.csv"), encoding="utf-8-sig")
# columns: model, AUC (str "0.910±0.006")
def parse_auc(s):
    return float(str(s).split("±")[0])
rnd_map = dict(zip(res["model"], res["AUC"].map(parse_auc)))

# careful: CSV stores "LR","RF","XGBoost" (v3 script uses those keys)
models = ["LR", "RF", "XGBoost"]
rnd_auc = [rnd_map[m] for m in models]

# spatial (grid snake, 300m buffer) from rigorous CV json
cv = json.load(open(os.path.join(OUT, "cv_rigorous.json"), encoding="utf-8"))
spa_map = cv["scheme_B_grid_snake_300m"]["results"]  # keys: LR, RF, XGBoost
spa_auc = [float(str(spa_map[m]["AUC"]).split("±")[0]) for m in models]

print("random:", {m: rnd_map[m] for m in models})
print("spatial:", {m: float(str(spa_map[m]["AUC"]).split("±")[0]) for m in models})

x = np.arange(len(models))
w = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
b1 = ax.bar(x - w/2, rnd_auc, w, label="Random 5-fold CV", color="#0072B2")
b2 = ax.bar(x + w/2, spa_auc, w, label="Spatial (grid binned + 300 m buffer)", color="#D55E00")
ax.set_ylabel("AUC")
ax.set_title("Model AUC: Random vs. Spatial Cross-Validation")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.legend(loc="lower right")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0.7, 1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_cv_comparison.png"), dpi=300)
plt.close(fig)
print("fig6_cv_comparison.png saved (v2.1, data-driven)")