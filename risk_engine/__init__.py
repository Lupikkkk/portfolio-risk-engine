"""Portfolio Risk Engine: VaR / Expected Shortfall, backtesting, stress testing and risk attribution."""

from .data import load_prices, simulate_prices, to_returns
from .portfolio import Portfolio
from .var import historical_var, parametric_var, monte_carlo_var, expected_shortfall
from .backtest import (rolling_var, kupiec_pof, christoffersen_independence, conditional_coverage,
                       es_backtest_z2)
from .garch import ewma_vol, fit_garch, fhs_var, rolling_fhs
from .stress import HISTORICAL_SCENARIOS, run_historical_stress, run_hypothetical_shock

__all__ = [
    "load_prices",
    "simulate_prices",
    "to_returns",
    "Portfolio",
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "expected_shortfall",
    "rolling_var",
    "kupiec_pof",
    "christoffersen_independence",
    "conditional_coverage",
    "es_backtest_z2",
    "ewma_vol",
    "fit_garch",
    "fhs_var",
    "rolling_fhs",
    "HISTORICAL_SCENARIOS",
    "run_historical_stress",
    "run_hypothetical_shock",
]
