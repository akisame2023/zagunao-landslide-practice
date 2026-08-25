"""
Stage 21b (P1 fixed): significance tests, correct implementation.
- Collect OOF prob/pred via fold loop (verified shapes)
- DeLong: standard formulas (rank all pooled scores together)
- McNemar, Brier, and calibration-in-the-large
"""
import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy import stats

OUT = r"C:\Users\Dell\Downloads\rsch1\experiment\outputs"
SEED = 42
FACTORS = ["elev", "slope", "aspect_sin", "aspect_cos", "plan_c", "prof_c",
           "twi", "spi", "sti", "relief", "ndvi"]
FEATURES = FACTORS + ["dist_stream"]

samples = pd.read_csv(os.path.join(OUT, "samples_raw2.csv"))
with rasterio.open(os.path.join(OUT, "elev.tif")) as s0:
    tf = s0.transform
with rasterio.open(os.path.join(OUT, "dist_stream_low.tif")) as s1:
    tf_low = s1.transform
rasters = {f: rasterio.open(os.path.join(OUT, f + ".tif")) for f in FACTORS}
rasters["dist_stream"] = rasterio.open(os.path.join(OUT, "dist_stream_low.tif"))
lon = samples["lon"].values; lat = samples["lat"].values

def extract(src, tr, lo, la):
    arr = src.read(1)
    c = (lo - tr.c) / tr.a; r = (tr.f - la) / (-tr.e)
    vals = []
    for ci, ri in zip(c, r):
        ci_i, ri_i = int(round(ci)), int(round(ri))
        if 0 <= ri_i < arr.shape[0] and 0 <= ci_i < arr.shape[1]:
            vals.append(arr[ri_i, ci_i])
        else:
            vals.append(np.nan)
    return np.array(vals)

for f in FACTORS:
    tr_use = tf_low if f == "twi" else tf
    samples[f] = extract(rasters[f], tr_use, lon, lat)
samples["dist_stream"] = extract(rasters["dist_stream"], tf_low, lon, lat)
for s in rasters.values():
    s.close()

df = samples.dropna().reset_index(drop=True)
pos = df[df.label == 1]; neg = df[df.label == 0]
n = min(len(pos), len(neg))
df = pd.concat([pos.iloc[:n], neg.sample(n, random_state=SEED)], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
X = df[FEATURES].values; y = df["label"].values
print("X,y shapes:", X.shape, y.shape, "balance:", np.bincount(y).tolist())

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

models = {
    "LR": LogisticRegression(max_iter=3000, random_state=SEED),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                 n_jobs=-1, random_state=SEED),
    "XGB": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=SEED, n_jobs=-1),
}
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)

# OOF collection
prob_oof = {m: np.zeros(len(y)) for m in models}
pred_oof = {m: np.zeros(len(y), dtype=int) for m in models}
for tr, te in CV.split(X, y):
    for name, mdl in models.items():
        Xtr, Xte = (Xs[tr], Xs[te]) if name == "LR" else (X[tr], X[te])
        mdl.fit(Xtr, y[tr])
        prob_oof[name][te] = mdl.predict_proba(Xte)[:, 1]
        pred_oof[name][te] = mdl.predict(Xte)

# sanity
for m in models:
    print(m, "OOF AUC:", round(roc_auc_score(y, prob_oof[m]), 4),
          "Brier:", round(brier_score_loss(y, prob_oof[m]), 4))

# ---- DeLong (correct, two-sample rank-based with covariance) ----
def auc_rank(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    # pooled ranks
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def delong_var(y, p1, p2):
    """Covariance of two AUC estimates (DeLong et al. 1988)."""
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    # sort by p1
    o1 = np.argsort(-p1)
    R1 = np.empty(n); R1[o1] = np.arange(1, n + 1)
    o2 = np.argsort(-p2)
    R2 = np.empty(n); R2[o2] = np.arange(1, n + 1)

    # structural components V10, V01 per classifier; V11 joint
    def comp(p, R):
        # V10: rank of positives minus i
        pos_idx = np.where(y == 1)[0]
        # sort positives by p desc
        pos_ord = pos_idx[np.argsort(-p[pos_idx])]
        ranks_pos = R[pos_ord]
        V10 = ((ranks_pos - (np.arange(n1) + 1)) ** 2).sum() / (n0 * n0) / n1
        # V01
        neg_idx = np.where(y == 0)[0]
        neg_ord = neg_idx[np.argsort(-p[neg_idx])]
        ranks_neg = R[neg_ord]
        V01 = ((ranks_neg - (np.arange(n0) + 1)) ** 2).sum() / (n1 * n1) / n0
        return V10, V01

    V10_1, V01_1 = comp(p1, R1)
    V10_2, V01_2 = comp(p2, R2)
    # V11: cross covariance over positive ranks
    pos_idx = np.where(y == 1)[0]
    po1 = pos_idx[np.argsort(-p1[pos_idx])]
    po2 = pos_idx[np.argsort(-p2[pos_idx])]
    R1p = R1[po1]; R2p = R2[po2]
    V11_pos = ((R1p - (np.arange(n1) + 1)) * (R2p - (np.arange(n1) + 1))).sum() / (n1 * n0 * n0) / n1
    neg_idx = np.where(y == 0)[0]
    ng1 = neg_idx[np.argsort(-p1[neg_idx])]
    ng2 = neg_idx[np.argsort(-p2[neg_idx])]
    R1n = R1[ng1]; R2n = R2[ng2]
    V11_neg = ((R1n - (np.arange(n0) + 1)) * (R2n - (np.arange(n0) + 1))).sum() / (n1 * n1 * n0) / n0
    var1 = V10_1 + V01_1
    var2 = V10_2 + V01_2
    cov = V11_pos + V11_neg
    return var1, var2, cov

def delong_test(y, p1, p2):
    a1 = auc_rank(y, p1); a2 = auc_rank(y, p2)
    v1, v2, cov = delong_var(y, p1, p2)
    se = np.sqrt(max(v1 + v2 - 2 * cov, 1e-12))
    z = (a1 - a2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return a1, a2, z, p

rows = []
print("\n=== DeLong tests ===")
for a, b in [("XGB", "LR"), ("XGB", "RF"), ("RF", "LR")]:
    a1, a2, z, p = delong_test(y, prob_oof[a], prob_oof[b])
    rows.append({"comparison": f"{a} vs {b}", "auc1": round(a1, 4), "auc2": round(a2, 4),
                 "z": round(z, 2), "p_delong": p})
    print(f"{a} vs {b}: {a1:.4f} vs {a2:.4f} z={z:.2f} p={p:.4g}")

print("\n=== McNemar ===")
for a, b in [("XGB", "LR"), ("XGB", "RF")]:
    n01 = int(((pred_oof[a] != y) & (pred_oof[b] == y)).sum())
    n10 = int(((pred_oof[a] == y) & (pred_oof[b] != y)).sum())
    chi2 = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    p = 1 - stats.chi2.cdf(chi2, 1)
    rows.append({"comparison": f"{a} vs {b}", "n01": n01, "n10": n10, "p_mcnemar": p})
    print(f"{a} vs {b}: n01={n01} n10={n10} p={p:.4g}")

print("\n=== Brier ===")
for m in models:
    b = brier_score_loss(y, prob_oof[m])
    rows.append({"comparison": f"{m} brier", "brier": round(b, 4)})
    print(f"{m}: {b:.4f}")

# calibration-in-the-large: mean predicted prob vs prevalence
print("\n=== calibration-in-the-large ===")
for m in models:
    meanp = prob_oof[m].mean()
    rows.append({"comparison": f"{m} cal", "mean_pred": round(meanp, 4), "prev": round(y.mean(), 4)})
    print(f"{m}: mean pred={meanp:.4f} vs prevalence={y.mean():.4f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "significance_tests_fixed.csv"), index=False, encoding="utf-8-sig")
print("\nsaved significance_tests_fixed.csv")