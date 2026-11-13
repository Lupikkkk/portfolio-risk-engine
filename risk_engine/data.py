"""Market data loading.

Prices come from Yahoo Finance (via ``yfinance``) or from a local CSV.
A synthetic generator is included so the engine and its tests run offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices (index = date, columns = tickers).

    If ``csv_path`` is given, prices are read from that file instead of the internet.
    """
    if csv_path is not None:
        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return prices[tickers].dropna()

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Install yfinance (`pip install yfinance`) or pass csv_path=...") from exc

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"No price data downloaded for {missing}; check the tickers or retry.")
    prices = prices[tickers].dropna()
    if prices.empty:
        raise RuntimeError("Downloaded prices have no dates in common across all tickers.")
    return prices


def simulate_prices(
    tickers: list[str],
    n_days: int = 2000,
    seed: int = 42,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """Simulate correlated prices with fat tails (Student-t shocks) and volatility clustering.

    Useful for offline demos and unit tests. Parameters are illustrative, not calibrated.
    """
    rng = np.random.default_rng(seed)
    k = len(tickers)

    # Random but well-conditioned correlation matrix
    a = rng.normal(size=(k, k))
    cov = a @ a.T + k * np.eye(k)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    chol = np.linalg.cholesky(corr)

    base_vol = rng.uniform(0.008, 0.02, size=k)  # daily vols
    drift = rng.uniform(0.0001, 0.0005, size=k)

    # GARCH(1,1)-style volatility multiplier shared across assets
    omega, alpha, beta = 0.05, 0.10, 0.85
    sigma2 = np.ones(n_days)
    shocks = rng.standard_t(df=4, size=(n_days, k)) / np.sqrt(2.0)  # t(4) has var 2
    returns = np.empty((n_days, k))
    for t in range(n_days):
        if t > 0:
            sigma2[t] = omega + alpha * (returns[t - 1] ** 2).mean() / (base_vol.mean() ** 2) + beta * sigma2[t - 1]
        z = chol @ shocks[t]
        returns[t] = drift + base_vol * np.sqrt(sigma2[t]) * z

    dates = pd.bdate_range(start=start, periods=n_days)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Daily returns from prices (simple by default, log if ``log=True``)."""
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()
