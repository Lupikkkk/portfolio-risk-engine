"""Conditional volatility models and Filtered Historical Simulation (FHS).

* EWMA (RiskMetrics) volatility
* GARCH(1,1) fitted by maximum likelihood (implemented from scratch with scipy)
* GJR-GARCH(1,1) (Glosten, Jagannathan & Runkle, 1993): negative shocks raise volatility by
  an extra gamma * e^2, the leverage effect seen in equity markets
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


def _filter_variance(news: np.ndarray, omega: float, beta: float, s0: float) -> np.ndarray:
    """s2_t = omega + news_{t-1} + beta * s2_{t-1}, with s2_0 = s0 (vectorised).

    ``news`` is the ARCH term: alpha * e^2 for GARCH, (alpha + gamma * 1[e < 0]) * e^2 for GJR."""
    if len(news) == 1:
        return np.array([s0])
    x = omega + news[:-1]
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
    gamma: float = 0.0  # GJR leverage term (0 for plain GARCH)

    @property
    def model(self) -> str:
        return "gjr" if self.gamma > 0 else "garch"

    @property
    def persistence(self) -> float:
        """alpha + gamma / 2 + beta (with symmetric shocks, half of them are negative)."""
        return self.alpha + self.gamma / 2 + self.beta

    @property
    def n_params(self) -> int:
        return 5 if self.model == "gjr" else 4

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * np.log(len(self.resid)) - 2 * self.loglik

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.omega / max(1 - self.persistence, 1e-12)))

    @property
    def half_life_days(self) -> float:
        """Days for a volatility shock to decay by half."""
        return float(np.log(0.5) / np.log(self.persistence)) if 0 < self.persistence < 1 else np.inf


def _apply(r: np.ndarray, mu: float, omega: float, alpha: float, beta: float, gamma: float = 0.0):
    e = r - mu
    news = (alpha + gamma * (e < 0)) * e ** 2
    s2 = _filter_variance(news, omega, beta, e.var())
    nxt = omega + news[-1] + beta * s2[-1]
    return e, s2, nxt


def fit_garch(returns: pd.Series | np.ndarray, model: str = "garch") -> GarchFit:
    """Gaussian (quasi-)maximum-likelihood GARCH(1,1) or GJR-GARCH(1,1).

    GARCH: s2_t = omega + alpha * e2_{t-1} + beta * s2_{t-1}
    GJR:   s2_t = omega + (alpha + gamma * 1[e_{t-1} < 0]) * e2_{t-1} + beta * s2_{t-1}
    Constraints: omega > 0, alpha, beta, gamma >= 0, alpha + gamma / 2 + beta < 1.
    """
    if model not in ("garch", "gjr"):
        raise ValueError("model must be 'garch' or 'gjr'")
    r = np.asarray(pd.Series(returns).dropna(), dtype=float) * SCALE
    mu = r.mean()
    var = r.var()
    gjr = model == "gjr"

    def nll(p):
        omega, alpha, beta = p[:3]
        e, s2, _ = _apply(r, mu, omega, alpha, beta, p[3] if gjr else 0.0)
        return 0.5 * np.sum(np.log(s2) + e ** 2 / s2)

    if gjr:
        starts = ((0.02, 0.90, 0.10), (0.05, 0.85, 0.12), (0.00, 0.92, 0.14))
        bounds = [(1e-8, None), (0.0, 0.999), (1e-6, 0.999), (0.0, 0.999)]
        stationary = {"type": "ineq", "fun": lambda p: 0.9995 - p[1] - p[2] - p[3] / 2}
    else:
        starts = ((0.08, 0.90, 0.0), (0.05, 0.93, 0.0), (0.15, 0.75, 0.0))
        bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]
        stationary = {"type": "ineq", "fun": lambda p: 0.9995 - p[1] - p[2]}

    best = None
    for a0, b0, g0 in starts:
        x0 = [var * (1 - a0 - b0 - g0 / 2), a0, b0] + ([g0] if gjr else [])
        res = minimize(nll, x0=x0, method="SLSQP", bounds=bounds, constraints=[stationary])
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = best.x[:3]
    gamma = float(best.x[3]) if gjr else 0.0
    e, s2, nxt = _apply(r, mu, omega, alpha, beta, gamma)
    sigma = np.sqrt(s2)
    loglik = -best.fun - 0.5 * len(r) * np.log(2 * np.pi)
    return GarchFit(mu / SCALE, omega / SCALE ** 2, alpha, beta, float(loglik),
                    sigma / SCALE, e / sigma, float(np.sqrt(nxt)) / SCALE, gamma)


def fhs_var(returns: pd.Series, alpha: float = 0.99, fit: GarchFit | None = None) -> tuple[float, float]:
    """One-day FHS VaR and ES (positive losses) for the day after the sample."""
    fit = fit or fit_garch(returns)
    q = np.quantile(fit.resid, 1 - alpha)
    tail = fit.resid[fit.resid <= q]
    var = -(fit.mu + fit.next_sigma * q)
    es = -(fit.mu + fit.next_sigma * tail.mean())
    return float(var), float(es)


def rolling_fhs(returns: pd.Series, alpha: float = 0.99, window: int = 750, refit: int = 20,
                model: str = "garch") -> pd.DataFrame:
    """Out-of-sample FHS forecasts. GARCH (or GJR-GARCH, ``model='gjr'``) parameters are
    re-estimated every ``refit`` days on the previous ``window`` days; in between, volatility is
    updated daily with fixed parameters. Each forecast uses only data available before that day."""
    r = returns.dropna()
    x = r.values
    rows = []
    params = None
    for i in range(window, len(x)):
        hist = x[i - window:i]
        if params is None or (i - window) % refit == 0:
            f = fit_garch(hist, model)
            params = (f.mu, f.omega, f.alpha, f.beta, f.gamma)
        mu, omega, a, b, g = params
        e, s2, nxt = _apply(hist, mu, omega, a, b, g)
        z = e / np.sqrt(s2)
        q = np.quantile(z, 1 - alpha)
        sig = np.sqrt(nxt)
        rows.append((-(mu + sig * q), -(mu + sig * z[z <= q].mean()), sig))
    out = pd.DataFrame(rows, columns=["var", "es", "sigma"], index=r.index[window:])
    out.insert(0, "return", x[window:])
    out["exception"] = out["return"] < -out["var"]
    return out
