"""v2 forecasters on the full-year dataset.

Same model families and hyperparameters as the original scripts 06/08, run over
the D5-corrected dataset so forecasts exist for all 8,760 hours of the test
year rather than the 4,081 daylight hours the original produced.

Also fits quantile forecasts (PV P10, load P90) for the Phase 6 reserve policy,
so that reserve can be held as battery headroom rather than bought with diesel.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

from . import paths as P

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RANDOM_SEED = 42

FEATURES_PV = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend",
    "PV_lag1", "PV_lag2", "PV_lag3", "PV_lag24", "PV_lag48", "PV_lag168",
    "GHI_lag1", "GHI_lag24", "GHI_lag48", "T2M_lag1", "T2M_lag24",
    "PV_roll6_mean", "PV_roll24_mean", "PV_roll48_mean",
    "GHI_roll6_mean", "GHI_roll24_mean",
    "clearsky_index_lag1", "clearsky_index_lag24",
]
FEATURES_LOAD = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "is_weekend",
    "load_lag1", "load_lag2", "load_lag3", "load_lag24", "load_lag48",
    "load_lag168", "load_roll6_mean", "load_roll6_std", "load_roll24_mean",
    "load_roll24_std", "load_roll48_mean",
]

XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=7,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
                  n_jobs=-1, early_stopping_rounds=50, eval_metric="rmse")
RF_PARAMS = dict(n_estimators=300, max_depth=18, min_samples_leaf=2,
                 random_state=RANDOM_SEED, n_jobs=-1)


def metrics(y, yhat):
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    return {
        "r2": float(r2_score(y, yhat)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, yhat)),
        "nrmse_pct": float(rmse / (y.max() - y.min()) * 100),
        "nmae_pct": float(mean_absolute_error(y, yhat) / y.mean() * 100),
    }


def load_splits():
    return (pd.read_csv(P.data("train.csv"), parse_dates=["datetime"]),
            pd.read_csv(P.data("validation.csv"), parse_dates=["datetime"]),
            pd.read_csv(P.data("test.csv"), parse_dates=["datetime"]))


def main(seed=RANDOM_SEED):
    train, val, test = load_splits()
    out = test[["datetime"]].copy()
    all_metrics, errors = {}, {}

    for target, feats, short in (("target_PV_kW", FEATURES_PV, "PV"),
                                 ("target_load_kW", FEATURES_LOAD, "load")):
        Xtr, ytr = train[feats].values, train[target].values
        Xv, yv = val[feats].values, val[target].values
        Xte, yte = test[feats].values, test[target].values

        p = dict(XGB_PARAMS, random_state=seed)
        m = xgb.XGBRegressor(**p)
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        pred = np.clip(m.predict(Xte), 0, None)
        all_metrics[f"xgb_{short}"] = metrics(yte, pred)
        out[f"{short}_actual"] = yte
        out[f"xgb_{short}_pred"] = pred
        errors[f"xgb_{short}"] = yte - pred
        print(f"  XGBoost {short:<5} R2={all_metrics[f'xgb_{short}']['r2']:.4f} "
              f"RMSE={all_metrics[f'xgb_{short}']['rmse']:.3f} "
              f"nRMSE={all_metrics[f'xgb_{short}']['nrmse_pct']:.2f}%")

        rf = RandomForestRegressor(**dict(RF_PARAMS, random_state=seed))
        rf.fit(Xtr, ytr)
        rpred = np.clip(rf.predict(Xte), 0, None)
        all_metrics[f"rf_{short}"] = metrics(yte, rpred)
        out[f"rf_{short}_pred"] = rpred
        errors[f"rf_{short}"] = yte - rpred
        print(f"  RF      {short:<5} R2={all_metrics[f'rf_{short}']['r2']:.4f} "
              f"RMSE={all_metrics[f'rf_{short}']['rmse']:.3f} "
              f"nRMSE={all_metrics[f'rf_{short}']['nrmse_pct']:.2f}%")

        # Quantile forecasts: PV P10 (conservative supply), load P90 (conservative demand)
        q = 0.10 if short == "PV" else 0.90
        qp = dict(XGB_PARAMS, random_state=seed, objective="reg:quantileerror",
                  quantile_alpha=q)
        qp.pop("eval_metric", None)
        qm = xgb.XGBRegressor(**qp)
        qm.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        out[f"xgb_{short}_q{int(q * 100)}"] = np.clip(qm.predict(Xte), 0, None)
        print(f"  XGB-q{int(q * 100):<3}{short:<5} fitted")

    out.to_csv(P.data("forecasts.csv"), index=False)
    np.savez(P.data("forecast_errors.npz"), **errors)
    (P.data("forecast_metrics.json")).write_text(json.dumps(all_metrics, indent=2))
    print(f"\nSaved v2_forecasts.csv ({len(out)} hours)")
    return all_metrics


if __name__ == "__main__":
    main()
