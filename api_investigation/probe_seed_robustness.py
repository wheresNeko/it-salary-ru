# -*- coding: utf-8 -*-
"""
Is "the neural network does not beat gradient boosting" robust to the seed?

The orphaned processes that were hogging the CPU could only have affected
wall-clock time, not results -- the pipeline is seeded. But that reasoning is
worth testing rather than asserting, and the honest question is a different
one: is the M2-vs-M3 gap larger than the seed-to-seed noise of the MLP?

Runs M2 once (deterministic) and M3 under several seeds, on the same split and
the same matrix.
"""
import os
import pathlib
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# This probe lives in api_investigation/; train_models.py is one level up.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import train_models as tm

df = tm.load()
data = df[df[tm.TARGET].notna()].copy()
train, test, cutoff = tm.temporal_split(data, df)
y_tr = train[tm.TARGET].to_numpy()
y_te = test[tm.TARGET].to_numpy()
median_rur = float(np.expm1(np.median(y_te)))
print(f"split at {cutoff.date()}: train {len(train):,} / test {len(test):,}")
print(f"test median salary: {median_rur:,.0f} RUR\n")

lin = tm.linear_matrix()
lin.fit(train)
Dtr, Dte, svd_var, _ = tm.build_design_b(
    train, test, lin.named_transformers_["text"])
imp, sc = SimpleImputer(strategy="median"), StandardScaler()
Atr = sc.fit_transform(imp.fit_transform(Dtr))
Ate = sc.transform(imp.transform(Dte))

# ---- M2: deterministic given the seed -------------------------------------
hgb = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
    min_samples_leaf=20, l2_regularization=1.0, random_state=tm.RANDOM_STATE)
hgb.fit(Atr, y_tr)
m2 = tm.score(y_te, hgb.predict(Ate))
print(f"M2 gradient boosting  R2={m2['r2_log']:.4f}  MAE={m2['mae_rur']:,.0f} "
      f"({100*m2['mae_rur']/median_rur:.1f}%)")

# ---- M3 across seeds -------------------------------------------------------
rows = []
for seed in (0, 1, 7, 42, 123, 2024, 99999):
    pred, info = tm.train_torch_mlp(Atr, y_tr, Ate, seed=seed)
    s = tm.score(y_te, pred)
    rows.append(s)
    print(f"M3 seed={seed:<6} R2={s['r2_log']:.4f}  MAE={s['mae_rur']:,.0f} "
          f"({100*s['mae_rur']/median_rur:.1f}%)  epochs={info['epochs_run']}")

r2 = np.array([r["r2_log"] for r in rows])
mae = np.array([r["mae_rur"] for r in rows])
print()
print(f"M3 across {len(rows)} seeds:")
print(f"  R2   min={r2.min():.4f}  max={r2.max():.4f}  mean={r2.mean():.4f}  "
      f"sd={r2.std(ddof=1):.4f}")
print(f"  MAE  min={mae.min():,.0f}  max={mae.max():,.0f}  mean={mae.mean():,.0f}")
print()
print(f"M2 R2={m2['r2_log']:.4f} vs M3 best-seed R2={r2.max():.4f}  -> "
      f"{'M2 still better' if m2['r2_log'] > r2.max() else 'M3 beats M2 on some seed'}")
print(f"M2 R2 beats {(r2 < m2['r2_log']).sum()}/{len(r2)} MLP seeds")
print(f"M2 MAE={m2['mae_rur']:,.0f} vs M3 best MAE={mae.min():,.0f}  -> "
      f"{'M2 still better' if m2['mae_rur'] < mae.min() else 'M3 beats M2 on some seed'}")
