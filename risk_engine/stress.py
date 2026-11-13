"""Stress testing: historical replay of crisis windows and hypothetical factor shocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Historical windows (start, end). The engine replays the portfolio's actual returns
# over each window, so results reflect real co-movement during the crisis.
HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "GFC: Lehman collapse (Sep-Nov 2008)": ("2008-09-01", "2008-11-20"),
    "Euro sovereign crisis (Jul-Sep 2011)": ("2011-07-01", "2011-09-30"),
    "China devaluation (Aug 2015)": ("2015-08-10", "2015-08-31"),
    "Volmageddon (Feb 2018)": ("2018-01-26", "2018-02-08"),
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rates shock (Jan-Oct 2022)": ("2022-01-03", "2022-10-12"),
    "SVB / regional banks (Mar 2023)": ("2023-03-08", "2023-03-17"),
}


def run_historical_stress(
    asset_returns: pd.DataFrame, weights: pd.Series, value: float = 1_000_000.0,
    scenarios: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Cumulative portfolio P&L over each historical window covered by the data."""
    scenarios = scenarios or HISTORICAL_SCENARIOS
    w = weights.reindex(asset_returns.columns).fillna(0.0)
    rows = []
    for name, (start, end) in scenarios.items():
        window = asset_returns.loc[start:end]
        if len(window) < 3:
            continue  # not covered by the data
        port = window @ w
        cum = float((1 + port).prod() - 1)
        path = (1 + port).cumprod()
        worst = float((path / path.cummax() - 1).min())
        rows.append({"scenario": name, "days": len(window), "portfolio_return": cum,
                     "pnl": cum * value, "max_drawdown_in_window": worst})
    return pd.DataFrame(rows).set_index("scenario") if rows else pd.DataFrame()


def run_hypothetical_shock(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    shocks: dict[str, float],
    value: float = 1_000_000.0,
) -> pd.DataFrame:
    """Apply instantaneous shocks to selected 'driver' assets and propagate them to the rest
    through regression betas estimated on history (conditional expectation under joint normality).

    Example: ``{"SPY": -0.20}`` -> equity market falls 20%; other assets move by beta * shock.
    """
    drivers = list(shocks)
    shock_vec = pd.Series(shocks)

    cov = asset_returns.cov()
    cov_xx = cov.loc[drivers, drivers].values
    cov_yx = cov.loc[:, drivers].values
    betas = pd.DataFrame(cov_yx @ np.linalg.inv(cov_xx), index=asset_returns.columns, columns=drivers)
    implied = betas @ shock_vec
    implied.loc[drivers] = shock_vec  # drivers move exactly by the shock

    w = weights.reindex(asset_returns.columns).fillna(0.0)
    out = pd.DataFrame({"weight": w, "implied_move": implied, "pnl": w * implied * value})
    out.loc["TOTAL"] = [w.sum(), float((w * implied).sum()), float((w * implied).sum() * value)]
    return out
