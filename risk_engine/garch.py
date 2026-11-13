"""Conditional volatility models and Filtered Historical Simulation (FHS).

* EWMA (RiskMetrics) volatility
* GARCH(1,1) fitted by maximum likelihood (implemented from scratch with scipy)
* FHS VaR / ES (Barone-Adesi, Giannopoulos & Vosper, 1999): standardise returns by their
  GARCH volatility, take the empirical quantile of the residuals, and rescale by the
  volatility forecast for tomorrow. This keeps the fat tails of historical simulation
  while reacting quickly to changes in volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

SCALE = 100.0  # returns are fitted in percent for numerical stability


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility: s2_t = lam * s2_{t-1} + (1 - lam) * r_{t-1}^2."""
    r = returns.dropna()
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)).dropna()


def _filter_variance(e2: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
    """s2_t = omega + alpha * e2_{t-1} + beta * s2_{t-1}, with s2_0 = s0 (vectorised)."""
    if len(e2) == 1:
        return np.array([s0])
    x = omega + alpha * e2[:-1]
    tail, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])
    return np.concatenate([[s0], tail])


@dataclass
class GarchFit:
    mu: float
    omega: float
    alpha: float
    beta: float
    loglik: float
    sigma: np.ndarray  # in-sample conditional volatility (return units)
    resid: np.ndarray  # standardised residuals z_t = (r_t - mu) / sigma_t
    next_sigma: float  # one-step-ahead volatility forecast

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.omega / max(1 - self.persistence, 1e-12)))

    @property
    def half_life_days(self) -> float:
        """Days for a volatility shock to decay by half."""
        return float(np.log(0.5) / np.log(self.persistence)) if 0 < self.persistence < 1 else np.inf


def _apply(r: np.ndarray, mu: float, omega: float, alpha: float, beta: float):
    e = r - mu
    s2 = _filter_variance(e ** 2, omega, alpha, beta, e.var())
    nxt = omega + alpha * e[-1] ** 2 + beta * s2[-1]
    return e, s2, nxt


def fit_garch(returns: pd.Series | np.ndarray) -> GarchFit:
    """Gaussian (quasi-)maximum-likelihood GARCH(1,1) with constraints
    omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float) * SCALE
    mu = r.mean()
    var = r.var()

    def nll(p):
        omega, alpha, beta = p
        e, s2, _ = _apply(r, mu, omega, alpha, beta)
        return 0.5 * np.sum(np.log(s2) + e ** 2 / s2)

    best = None
    for a0, b0 in ((0.08, 0.90), (0.05, 0.93), (0.15, 0.75)):
        res = minimize(
            nll, x0=[var * (1 - a0 - b0), a0, b0], method="SLSQP",
            bounds=[(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)],
            constraints=[{"type": "ineq", "fun": lambda p: 0.9995 - p[1] - p[2]}],
        )
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = best.x
    e, s2, nxt = _apply(r, mu, omega, alpha, beta)
    sigma = np.sqrt(s2)
    loglik = -best.fun - 0.5 * len(r) * np.log(2 * np.pi)
    return GarchFit(mu / SCALE, omega / SCALE ** 2, alpha, beta, float(loglik),
                    sigma / SCALE, e / sigma, float(np.sqrt(nxt)) / SCALE)


def fhs_var(returns: pd.Series, alpha: float = 0.99, fit: GarchFit | None = None) -> tuple[float, float]:
    """One-day FHS VaR and ES (positive losses) for the day after the sample."""
    fit = fit or fit_garch(returns)
    q = np.quantile(fit.resid, 1 - alpha)
    tail = fit.resid[fit.resid <= q]
    var = -(fit.mu + fit.next_sigma * q)
    es = -(fit.mu + fit.next_sigma * tail.mean())
    return float(var), float(es)


def rolling_fhs(returns: pd.Series, alpha: float = 0.99, window: int = 750, refit: int = 20) -> pd.DataFrame:
    """Out-of-sample FHS forecasts. GARCH parameters are re-estimated every ``refit`` days on
    the previous ``window`` days; in between, volatility is updated daily with fixed parameters.
    Each forecast uses only data available before that day."""
    r = returns.dropna()
    x = r.values
    rows = []
    params = None
    for i in range(window, len(x)):
        hist = x[i - window:i]
        if params is None or (i - window) % refit == 0:
            f = fit_garch(hist)
            params = (f.mu, f.omega, f.alpha, f.beta)
        mu, omega, a, b = params
        e, s2, nxt = _apply(hist, mu, omega, a, b)
        z = e / np.sqrt(s2)
        q = np.quantile(z, 1 - alpha)
        sig = np.sqrt(nxt)
        rows.append((-(mu + sig * q), -(mu + sig * z[z <= q].mean()), sig))
    out = pd.DataFrame(rows, columns=["var", "es", "sigma"], index=r.index[window:])
    out.insert(0, "return", x[window:])
    out["exception"] = out["return"] < -out["var"]
    return out
