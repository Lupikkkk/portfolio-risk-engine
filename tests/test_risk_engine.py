import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk_engine import (
    Portfolio, christoffersen_independence, expected_shortfall, historical_var, kupiec_pof,
    monte_carlo_var, parametric_var, rolling_var, run_hypothetical_shock, simulate_prices, to_returns,
)


@pytest.fixture(scope="module")
def data():
    prices = simulate_prices(["A", "B", "C"], n_days=1500, seed=1)
    rets = to_returns(prices)
    w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    return rets, w


def test_parametric_var_matches_normal_formula():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 200_000))
    expected = -norm.ppf(0.01) * 0.01
    assert parametric_var(r, 0.99) == pytest.approx(expected, rel=0.02)
    assert historical_var(r, 0.99) == pytest.approx(expected, rel=0.03)


def test_es_exceeds_var(data):
    rets, w = data
    pr = rets @ w
    assert expected_shortfall(pr, 0.99) >= historical_var(pr, 0.99)


def test_var_monotone_in_confidence(data):
    rets, w = data
    pr = rets @ w
    assert historical_var(pr, 0.99) > historical_var(pr, 0.95)


def test_monte_carlo_normal_close_to_parametric(data):
    rets, w = data
    pr = rets @ w
    mc = monte_carlo_var(rets, w, 0.99, n_sims=200_000, dist="normal", seed=3)
    assert mc == pytest.approx(parametric_var(pr, 0.99), rel=0.05)


def test_component_var_sums_to_total(data):
    rets, w = data
    port = Portfolio(rets, w, value=1.0)
    cv = port.component_var(0.99)
    total = norm.ppf(0.99) * port.volatility()
    assert cv["component_var"].sum() == pytest.approx(total, rel=1e-9)
    assert cv["pct_of_total"].sum() == pytest.approx(1.0)


def test_kupiec_accepts_correct_rate():
    rng = np.random.default_rng(5)
    exc = pd.Series(rng.random(5000) < 0.01)
    assert not kupiec_pof(exc, 0.99)["reject_H0_5pct"]


def test_kupiec_rejects_bad_rate():
    rng = np.random.default_rng(6)
    exc = pd.Series(rng.random(2000) < 0.05)
    assert kupiec_pof(exc, 0.99)["reject_H0_5pct"]


def test_christoffersen_detects_clustering():
    e = np.zeros(1000, dtype=bool)
    for start in range(100, 1000, 200):
        e[start:start + 5] = True  # clustered exceptions
    assert christoffersen_independence(pd.Series(e))["reject_H0_5pct"]


def test_rolling_var_has_no_lookahead(data):
    rets, w = data
    pr = rets @ w
    bt = rolling_var(pr, 0.99, 250)
    first_day = bt.index[0]
    assert bt.loc[first_day, "var"] == pytest.approx(historical_var(pr.loc[:first_day].iloc[:-1], 0.99))


def test_hypothetical_shock_driver_moves_exactly(data):
    rets, w = data
    out = run_hypothetical_shock(rets, w, {"A": -0.2}, value=100)
    assert out.loc["A", "implied_move"] == pytest.approx(-0.2)


def _simulate_garch(n, omega, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return pd.Series(r)


def test_garch_recovers_true_parameters():
    from risk_engine import fit_garch
    fit = fit_garch(_simulate_garch(20_000, 2e-6, 0.10, 0.85, seed=11))
    assert fit.alpha == pytest.approx(0.10, abs=0.02)
    assert fit.beta == pytest.approx(0.85, abs=0.03)
    assert fit.long_run_vol == pytest.approx(np.sqrt(2e-6 / 0.05), rel=0.1)
    assert abs(np.std(fit.resid) - 1) < 0.02  # standardised residuals have unit variance


def test_fhs_reacts_to_volatility_but_historical_does_not():
    from risk_engine import fhs_var
    r = _simulate_garch(1500, 2e-6, 0.10, 0.85, seed=2)
    calm = fhs_var(r, 0.99)[0]
    shocked = r.copy()
    shocked.iloc[-1] = -0.08  # a crash on the last day
    assert fhs_var(shocked, 0.99)[0] > 2 * calm
    assert historical_var(shocked, 0.99) < 1.2 * historical_var(r, 0.99)


def test_fhs_fixes_exception_clustering():
    from risk_engine import rolling_fhs
    r = _simulate_garch(4000, 1e-6, 0.12, 0.86, seed=4)
    hs = rolling_var(r, 0.99, 250).iloc[500:]
    fhs = rolling_fhs(r, 0.99, 750, refit=50)
    p_hs = christoffersen_independence(hs["exception"])["p_value"]
    p_fhs = christoffersen_independence(fhs["exception"])["p_value"]
    assert p_fhs > 0.05 > p_hs


def test_es_backtest_z2_centred_at_zero_for_correct_model():
    from risk_engine import es_backtest_z2
    rng = np.random.default_rng(8)
    n, a = 100_000, 0.975
    r = pd.Series(rng.standard_normal(n))
    var = pd.Series(np.full(n, -norm.ppf(1 - a)))
    es = pd.Series(np.full(n, norm.pdf(norm.ppf(1 - a)) / (1 - a)))
    assert abs(es_backtest_z2(r, var, es, a)["Z2"]) < 0.05
    assert es_backtest_z2(r * 1.6, var, es, a)["reject_5pct"]  # true vol 60% higher than model


def test_student_t_var_recovers_true_quantile():
    from scipy.stats import t as student_t
    r = pd.Series(student_t.rvs(4, loc=0.0003, scale=0.008, size=50_000, random_state=7))
    true_var = -student_t.ppf(0.01, 4, 0.0003, 0.008)
    assert parametric_var(r, 0.99, dist="t") == pytest.approx(true_var, rel=0.05)
    assert parametric_var(r, 0.99, dist="t") > parametric_var(r, 0.99, dist="normal")  # fat tails


def test_student_t_es_matches_closed_form_and_simulation():
    from scipy.stats import t as student_t
    r = pd.Series(student_t.rvs(4, loc=0.0, scale=0.01, size=200_000, random_state=9))
    q = np.quantile(r, 0.025)
    empirical_es = -r[r <= q].mean()
    assert expected_shortfall(r, 0.975, "t") == pytest.approx(empirical_es, rel=0.05)
    assert expected_shortfall(r, 0.975, "t") > expected_shortfall(r, 0.975, "normal")  # fat tails


def test_kupiec_zero_exceptions_rejects_on_long_sample():
    res = kupiec_pof(pd.Series(np.zeros(1000, dtype=bool)), 0.99)
    assert res["LR_pof"] == pytest.approx(-2 * 1000 * np.log(0.99))
    assert res["reject_H0_5pct"]  # 0 exceptions when 10 are expected is too conservative


def test_portfolio_rejects_zero_net_weights(data):
    rets, _ = data
    with pytest.raises(ValueError):
        Portfolio(rets, pd.Series([1.0, -1.0, 0.0], index=["A", "B", "C"]))
