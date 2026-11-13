"""Portfolio object: weights, P&L series and risk attribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class Portfolio:
    returns: pd.DataFrame  # daily asset returns
    weights: pd.Series  # portfolio weights, indexed by ticker
    value: float = 1_000_000.0  # portfolio value in currency units

    def __post_init__(self) -> None:
        self.weights = self.weights.reindex(self.returns.columns).fillna(0.0)
        total = self.weights.sum()
        if np.isclose(total, 0.0):
            raise ValueError("weights sum to zero; pass weights that sum to a non-zero net exposure")
        if not np.isclose(total, 1.0):
            self.weights = self.weights / total

    @classmethod
    def equal_weight(cls, returns: pd.DataFrame, value: float = 1_000_000.0) -> "Portfolio":
        w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        return cls(returns, w, value)

    @property
    def pnl_returns(self) -> pd.Series:
        """Daily portfolio return series (constant weights, daily rebalanced)."""
        return self.returns @ self.weights

    @property
    def cov(self) -> pd.DataFrame:
        return self.returns.cov()

    def volatility(self, annualise: bool = False) -> float:
        vol = float(np.sqrt(self.weights @ self.cov @ self.weights))
        return vol * np.sqrt(252) if annualise else vol

    def component_var(self, alpha: float = 0.99) -> pd.DataFrame:
        """Euler decomposition of parametric (variance-covariance) VaR.

        Marginal VaR_i = z * (Sigma w)_i / sigma_p ; Component VaR_i = w_i * Marginal VaR_i.
        Components sum exactly to total parametric VaR (ignoring the mean).
        """
        z = norm.ppf(alpha)
        sigma_p = self.volatility()
        marginal = z * (self.cov @ self.weights) / sigma_p
        component = self.weights * marginal
        total = component.sum()
        out = pd.DataFrame(
            {
                "weight": self.weights,
                "marginal_var": marginal * self.value,
                "component_var": component * self.value,
                "pct_of_total": component / total,
            }
        )
        return out.sort_values("component_var", ascending=False)

    def summary(self) -> dict:
        r = self.pnl_returns
        ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        cum = (1 + r).cumprod()
        max_dd = float((cum / cum.cummax() - 1).min())
        return {
            "annual_return": float(ann_ret),
            "annual_volatility": float(ann_vol),
            "sharpe_rf0": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
            "max_drawdown": max_dd,
            "observations": int(len(r)),
        }
