"""Value-at-Risk and Expected Shortfall.

Conventions: VaR and ES are reported as POSITIVE losses for a one-day horizon,
scaled to ``horizon`` days with the square-root-of-time rule where relevant.
``alpha`` is the confidence level (e.g. 0.99).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


def historical_var(returns: pd.Series, alpha: float = 0.99, horizon: int = 1) -> float:
    """Historical simulation VaR: empirical (1 - alpha) quantile of returns."""
    q = np.quantile(returns.dropna(), 1 - alpha)
    return float(-q * np.sqrt(horizon))


def parametric_var(
    returns: pd.Series,
    alpha: float = 0.99,
    horizon: int = 1,
    dist: str = "normal",
    dof: float | None = None,
) -> float:
    """Variance-covariance VaR assuming normal or Student-t returns.

    With ``dof=None`` the Student-t is fitted by maximum likelihood (dof, location, scale).
    With a fixed ``dof`` the t quantile is rescaled to match the sample variance.
    """
    r = returns.dropna()
    mu, sigma = r.mean(), r.std()
    if dist == "normal":
        q = norm.ppf(1 - alpha)
        return float(-(mu * horizon + q * sigma * np.sqrt(horizon)))
    if dist == "t":
        if dof is None:
            # full maximum-likelihood fit of degrees of freedom, location and scale
            # sensible starting values matter: the default start can end in a poor local optimum
            dof, loc, scale = student_t.fit(r, 5, loc=mu, scale=sigma * 0.8)
            return float(-(loc * horizon + student_t.ppf(1 - alpha, dof) * scale * np.sqrt(horizon)))
        q = student_t.ppf(1 - alpha, dof) * np.sqrt((dof - 2) / dof)  # unit-variance t
        return float(-(mu * horizon + q * sigma * np.sqrt(horizon)))
    raise ValueError("dist must be 'normal' or 't'")


def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.99,
    horizon: int = 1,
    n_sims: int = 100_000,
    dist: str = "t",
    dof: float = 5.0,
    seed: int | None = 0,
    return_ES: bool = False,
):
    """Monte Carlo VaR (and optionally ES) with correlated multivariate shocks.

    ``dist='t'`` draws multivariate Student-t shocks (fat tails, tail dependence);
    ``dist='normal'`` draws multivariate normal shocks. Horizon returns are simulated directly.
    """
    rng = np.random.default_rng(seed)
    w = weights.reindex(asset_returns.columns).values
    mu = asset_returns.mean().values * horizon
    cov = asset_returns.cov().values * horizon
    chol = np.linalg.cholesky(cov)
    k = len(w)

    z = rng.standard_normal((n_sims, k))
    if dist == "t":
        chi = rng.chisquare(dof, size=(n_sims, 1))
        z = z / np.sqrt(chi / dof) * np.sqrt((dof - 2) / dof)  # unit variance
    elif dist != "normal":
        raise ValueError("dist must be 'normal' or 't'")

    sims = mu + z @ chol.T
    port = sims @ w
    var = -np.quantile(port, 1 - alpha)
    if return_ES:
        es = -port[port <= -var].mean()
        return float(var), float(es)
    return float(var)


def expected_shortfall(
    returns: pd.Series, alpha: float = 0.975, method: str = "historical", horizon: int = 1
) -> float:
    """Expected Shortfall (CVaR): average loss beyond VaR.

    Default alpha = 97.5%, the FRTB (Basel) standard.
    """
    r = returns.dropna()
    if method == "historical":
        var = historical_var(r, alpha)
        tail = r[r <= -var]
        return float(-tail.mean() * np.sqrt(horizon))
    if method == "normal":
        mu, sigma = r.mean(), r.std()
        z = norm.ppf(1 - alpha)
        es = -mu + sigma * norm.pdf(z) / (1 - alpha)
        return float(es * np.sqrt(horizon))
    if method == "t":
        # Student-t fitted by maximum likelihood; closed form for the tail mean:
        # ES = -loc + scale * f(q) / (1 - alpha) * (dof + q^2) / (dof - 1), q = t_dof^{-1}(1 - alpha)
        mu, sigma = r.mean(), r.std()
        dof, loc, scale = student_t.fit(r, 5, loc=mu, scale=sigma * 0.8)
        if dof <= 1:
            return float("inf")
        q = student_t.ppf(1 - alpha, dof)
        tail = student_t.pdf(q, dof) / (1 - alpha) * (dof + q ** 2) / (dof - 1)
        return float(-loc * horizon + scale * tail * np.sqrt(horizon))
    raise ValueError("method must be 'historical', 'normal' or 't'")


def var_table(returns: pd.Series, asset_returns: pd.DataFrame, weights: pd.Series, value: float = 1.0) -> pd.DataFrame:
    """Side-by-side comparison of methods at 95% / 99% (and ES 97.5%), in currency units."""
    rows = {}
    for a in (0.95, 0.99):
        rows[f"VaR {int(a * 100)}%"] = {
            "Historical": historical_var(returns, a),
            "Parametric (normal)": parametric_var(returns, a, dist="normal"),
            "Parametric (Student-t)": parametric_var(returns, a, dist="t"),
            "Monte Carlo (t, df=5)": monte_carlo_var(asset_returns, weights, a, n_sims=50_000),
        }
    rows["ES 97.5%"] = {
        "Historical": expected_shortfall(returns, 0.975, "historical"),
        "Parametric (normal)": expected_shortfall(returns, 0.975, "normal"),
        "Parametric (Student-t)": expected_shortfall(returns, 0.975, "t"),
        "Monte Carlo (t, df=5)": monte_carlo_var(asset_returns, weights, 0.975, n_sims=50_000, return_ES=True)[1],
    }
    from .garch import fit_garch, fhs_var

    fit = fit_garch(returns)
    for a, label in ((0.95, "VaR 95%"), (0.99, "VaR 99%")):
        rows[label]["FHS-GARCH (conditional)"] = fhs_var(returns, a, fit)[0]
    rows["ES 97.5%"]["FHS-GARCH (conditional)"] = fhs_var(returns, 0.975, fit)[1]
    return pd.DataFrame(rows).T * value
