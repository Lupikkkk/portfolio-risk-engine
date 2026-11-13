"""VaR backtesting: rolling out-of-sample forecasts plus Kupiec and Christoffersen tests,
and the Basel traffic-light zone."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .var import historical_var, parametric_var


def rolling_var(
    returns: pd.Series, alpha: float = 0.99, window: int = 250, method: str = "historical"
) -> pd.DataFrame:
    """One-day-ahead VaR forecasts using only the previous ``window`` observations.

    Returns a DataFrame with the realised return, the VaR forecast and an exception flag.
    """
    fn = {
        "historical": lambda x: historical_var(x, alpha),
        "normal": lambda x: parametric_var(x, alpha, dist="normal"),
        "t": lambda x: parametric_var(x, alpha, dist="t", dof=5),
    }[method]
    r = returns.dropna()
    forecasts = [fn(r.iloc[i - window : i]) for i in range(window, len(r))]
    out = pd.DataFrame({"return": r.iloc[window:].values, "var": forecasts}, index=r.index[window:])
    out["exception"] = out["return"] < -out["var"]
    return out


def kupiec_pof(exceptions: pd.Series, alpha: float = 0.99) -> dict:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: exception rate == 1 - alpha. LR ~ chi2(1).
    """
    n = int(len(exceptions))
    x = int(exceptions.sum())
    p = 1 - alpha
    phat = x / n if n else 0.0

    def _ll(prob: float) -> float:
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - prob) + x * np.log(prob)

    lr = -2 * (_ll(p) - _ll(phat))  # _ll clips phat, so x = 0 and x = n are handled
    pval = 1 - chi2.cdf(lr, 1)
    return {"n": n, "exceptions": x, "expected": n * p, "rate": phat, "LR_pof": lr, "p_value": pval,
            "reject_H0_5pct": pval < 0.05}


def christoffersen_independence(exceptions: pd.Series) -> dict:
    """Christoffersen (1998) test that exceptions are not clustered (first-order Markov).

    LR_ind ~ chi2(1).
    """
    e = exceptions.astype(int).values
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(e[:-1], e[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    def _safe_log(x: float) -> float:
        return np.log(x) if x > 0 else 0.0

    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (
        n00 * _safe_log(1 - pi0) + n01 * _safe_log(pi0) + n10 * _safe_log(1 - pi1) + n11 * _safe_log(pi1)
    )
    lr = -2 * (ll_null - ll_alt)
    pval = 1 - chi2.cdf(lr, 1)
    return {"pi_after_no_exception": pi0, "pi_after_exception": pi1, "LR_ind": lr, "p_value": pval,
            "reject_H0_5pct": pval < 0.05}


def basel_traffic_light(exceptions_last_250: int) -> str:
    """Basel Committee traffic-light zone for 99% VaR over 250 days."""
    if exceptions_last_250 <= 4:
        return "green"
    if exceptions_last_250 <= 9:
        return "yellow"
    return "red"


def conditional_coverage(exceptions: pd.Series, alpha: float = 0.99) -> dict:
    """Christoffersen conditional-coverage test: LR_cc = LR_pof + LR_ind ~ chi2(2).
    Rejects if the exception rate is wrong OR exceptions cluster."""
    lr = kupiec_pof(exceptions, alpha)["LR_pof"] + christoffersen_independence(exceptions)["LR_ind"]
    pval = 1 - chi2.cdf(lr, 2)
    return {"LR_cc": lr, "p_value": pval, "reject_H0_5pct": pval < 0.05}


def es_backtest_z2(returns: pd.Series, var: pd.Series, es: pd.Series, alpha: float = 0.975) -> dict:
    """Acerbi & Szekely (2014) unconditional ES backtest, statistic Z2.

    Z2 = sum_t( r_t * I_t / ((1 - alpha) * ES_t) ) / T + 1, where I_t = 1 on VaR exceptions.
    E[Z2] = 0 if the model is correct; Z2 < 0 means losses in the tail are larger or more
    frequent than predicted. The authors report a 5% critical value close to -0.7 across a
    wide range of distributions, which is used here as the rejection threshold.
    """
    r, v, e = returns.values, var.values, es.values
    hit = r < -v
    z2 = float(np.sum(r[hit] / ((1 - alpha) * e[hit])) / len(r) + 1)
    return {"Z2": z2, "exceptions": int(hit.sum()), "expected": len(r) * (1 - alpha),
            "reject_5pct": z2 < -0.7}


def backtest_report(returns: pd.Series, alpha: float = 0.99, window: int = 250,
                    fhs_window: int = 750, include_fhs: bool = True) -> pd.DataFrame:
    """Compare methods on the same out-of-sample period (all start after ``fhs_window`` days
    when FHS is included, so the comparison is like for like)."""
    from .garch import rolling_fhs

    runs = {m: rolling_var(returns, alpha, window, m) for m in ("historical", "normal", "t")}
    if include_fhs and len(returns.dropna()) > fhs_window + 250:
        runs["fhs_garch"] = rolling_fhs(returns, alpha, fhs_window)
        start = runs["fhs_garch"].index[0]
        runs = {m: bt.loc[start:] for m, bt in runs.items()}
    rows = {}
    for method, bt in runs.items():
        k = kupiec_pof(bt["exception"], alpha)
        c = christoffersen_independence(bt["exception"])
        cc = conditional_coverage(bt["exception"], alpha)
        last = int(bt["exception"].iloc[-250:].sum())
        rows[method] = {
            "exceptions": k["exceptions"],
            "expected": round(k["expected"], 1),
            "kupiec_p": round(k["p_value"], 4),
            "independence_p": round(c["p_value"], 4),
            "cond_coverage_p": round(cc["p_value"], 4),
            "last_250_exceptions": last,
            "basel_zone": basel_traffic_light(last),
        }
    return pd.DataFrame(rows).T
