from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.config import EvalSpec
from alpha_lab.types import EvaluationResult


def _forward_return(frame: pd.DataFrame, period: int) -> pd.Series:
    close = frame["close"].astype(float)
    by_symbol = close.groupby(level=1)
    return by_symbol.shift(-period) / close - 1.0


def _group_corr(a: pd.Series, b: pd.Series) -> pd.Series:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)

    def _corr(x: pd.DataFrame) -> float:
        if x["a"].nunique() < 2 or x["b"].nunique() < 2:
            return np.nan
        return float(x["a"].corr(x["b"]))

    return joined.groupby(level=0).apply(_corr)


def _group_rank_corr(a: pd.Series, b: pd.Series) -> pd.Series:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)

    def _spearman(x: pd.DataFrame) -> float:
        if x["a"].nunique() < 2 or x["b"].nunique() < 2:
            return np.nan
        return float(x["a"].rank().corr(x["b"].rank()))

    return joined.groupby(level=0).apply(_spearman)


def _make_quantiles(series: pd.Series, q: int) -> pd.Series:
    def _qcut(x: pd.Series) -> pd.Series:
        if x.notna().sum() < q:
            return pd.Series(np.nan, index=x.index)
        ranks = x.rank(method="first")
        return pd.qcut(ranks, q=q, labels=False, duplicates="drop")

    return series.groupby(level=0, group_keys=False).apply(_qcut)


def _turnover_by_bucket(bucket: pd.Series, target_bucket: int) -> pd.Series:
    groups = bucket.groupby(level=0)
    dates = sorted(groups.groups.keys())
    turnover = []
    prev_set: set[str] | None = None

    for dt in dates:
        idx = groups.groups[dt]
        current_symbols = set(
            bucket.loc[idx][bucket.loc[idx] == target_bucket].index.get_level_values(1).tolist()
        )
        if prev_set is None:
            turnover.append((dt, np.nan))
        else:
            union = prev_set | current_symbols
            inter = prev_set & current_symbols
            value = 0.0 if not union else 1.0 - len(inter) / len(union)
            turnover.append((dt, value))
        prev_set = current_symbols

    return pd.Series(dict(turnover), dtype=float)


def _calc_stability(ic_series: pd.Series, long_short: pd.Series) -> pd.DataFrame:
    if ic_series.empty:
        return pd.DataFrame(columns=["ic_mean", "ic_ir", "ls_mean", "ls_sharpe"])

    frame = pd.concat([ic_series.rename("ic"), long_short.rename("ls")], axis=1)
    frame.index = pd.to_datetime(frame.index)

    rows = []
    for year, chunk in frame.groupby(frame.index.year):
        ic_mean = float(chunk["ic"].mean())
        ic_std = float(chunk["ic"].std(ddof=0))
        ls_mean = float(chunk["ls"].mean())
        ls_std = float(chunk["ls"].std(ddof=0))
        ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
        ls_sharpe = (ls_mean / ls_std * math.sqrt(252)) if ls_std > 0 else np.nan
        rows.append(
            {
                "segment": str(year),
                "ic_mean": ic_mean,
                "ic_ir": ic_ir,
                "ls_mean": ls_mean,
                "ls_sharpe": ls_sharpe,
            }
        )
    return pd.DataFrame(rows)


def _walk_forward_ic(ic_series: pd.Series, train: int, test: int) -> dict[str, float]:
    values = ic_series.dropna().to_numpy(dtype=float)
    if len(values) < train + test:
        return {"wf_oos_ic_mean": float(np.nan), "wf_oos_ic_std": float(np.nan)}

    cursor = 0
    oos_means: list[float] = []
    while cursor + train + test <= len(values):
        oos = values[cursor + train : cursor + train + test]
        oos_means.append(float(np.mean(oos)))
        cursor += test
    if not oos_means:
        return {"wf_oos_ic_mean": float(np.nan), "wf_oos_ic_std": float(np.nan)}
    return {
        "wf_oos_ic_mean": float(np.mean(oos_means)),
        "wf_oos_ic_std": float(np.std(oos_means, ddof=0)),
    }


def _attribution(long_short: pd.Series, market_return: pd.Series) -> dict[str, float]:
    aligned = pd.concat(
        [long_short.rename("ls"), market_return.rename("mkt")],
        axis=1,
        sort=False,
    ).dropna()
    if len(aligned) < 10:
        return {"alpha": float(np.nan), "beta": float(np.nan), "r2": float(np.nan)}

    X = np.column_stack([np.ones(len(aligned)), aligned["mkt"].to_numpy()])
    y = aligned["ls"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(beta[0]), "beta": float(beta[1]), "r2": r2}


@dataclass(slots=True)
class Evaluator:
    spec: EvalSpec

    def evaluate(self, frame: pd.DataFrame, factor: pd.Series) -> EvaluationResult:
        fwd = _forward_return(frame, self.spec.forward_period)
        ic_series = _group_corr(factor, fwd)
        rank_ic_series = _group_rank_corr(factor, fwd)

        quantiles = _make_quantiles(factor, self.spec.quantiles)
        joined = pd.concat([quantiles.rename("q"), fwd.rename("ret")], axis=1).dropna()

        quantile_returns = (
            joined.groupby([joined.index.get_level_values(0), "q"])["ret"]
            .mean()
            .unstack("q")
            .sort_index()
        )

        top = quantile_returns.columns.max() if not quantile_returns.empty else 0
        bottom = quantile_returns.columns.min() if not quantile_returns.empty else 0
        long_short = (
            quantile_returns[top] - quantile_returns[bottom]
            if not quantile_returns.empty
            else pd.Series(dtype=float)
        )

        quantiles_non_na = quantiles.dropna()
        turnover = (
            _turnover_by_bucket(quantiles_non_na, int(top))
            if not quantiles_non_na.empty
            else pd.Series(dtype=float)
        )
        stability = _calc_stability(ic_series, long_short)
        market_return = fwd.groupby(level=0).mean()
        attribution = _attribution(long_short, market_return)
        walk_forward = _walk_forward_ic(
            ic_series,
            self.spec.walk_forward_train,
            self.spec.walk_forward_test,
        )

        metrics = {
            "ic_mean": float(ic_series.mean()),
            "ic_ir": float(ic_series.mean() / (ic_series.std(ddof=0) + 1e-12)),
            "rank_ic_mean": float(rank_ic_series.mean()),
            "ls_mean": float(long_short.mean()) if not long_short.empty else float("nan"),
            "ls_sharpe": (
                float(long_short.mean() / (long_short.std(ddof=0) + 1e-12) * np.sqrt(252))
                if not long_short.empty
                else float("nan")
            ),
            "turnover_mean": float(turnover.mean()) if not turnover.empty else float("nan"),
            **walk_forward,
            **attribution,
        }

        return EvaluationResult(
            metrics=metrics,
            ic_series=ic_series,
            rank_ic_series=rank_ic_series,
            quantile_returns=quantile_returns,
            long_short_returns=long_short,
            turnover=turnover,
            stability=stability,
            attribution=attribution,
        )
