"""Command-line risk report.

Examples
--------
Real data (needs internet + yfinance):
    python main.py --tickers SPY TLT GLD EFA QQQ --weights 0.35 0.25 0.10 0.15 0.15

Offline demo with simulated data:
    python main.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from risk_engine import Portfolio, load_prices, simulate_prices, to_returns
from risk_engine.backtest import backtest_report, es_backtest_z2, rolling_var
from risk_engine.garch import fit_garch, rolling_fhs
from risk_engine.stress import run_historical_stress, run_hypothetical_shock
from risk_engine.var import var_table

REPORTS = Path(__file__).parent / "reports"


def _fmt(df: pd.DataFrame, formats: dict[str, str], default: str = "{:,.0f}") -> str:
    """Plain-text table with a number format per column (used for reports/summary.md)."""
    out = df.copy().astype(object)
    for col in df.columns:
        f = formats.get(col, default)
        out[col] = [f.format(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else v
                    for v in df[col]]
    return out.to_string()


def main() -> None:
    p = argparse.ArgumentParser(description="Portfolio Risk Engine report")
    p.add_argument("--tickers", nargs="+", default=["SPY", "TLT", "GLD", "EFA", "QQQ"])
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--start", default="2007-01-01")
    p.add_argument("--value", type=float, default=1_000_000)
    p.add_argument("--csv", default=None, help="Local CSV of prices instead of downloading")
    p.add_argument("--demo", action="store_true", help="Use simulated data (offline)")
    args = p.parse_args()

    if args.demo:
        prices = simulate_prices(args.tickers, n_days=2500)
        print("NOTE: using SIMULATED prices (demo mode).\n")
    else:
        prices = load_prices(args.tickers, start=args.start, csv_path=args.csv)

    rets = to_returns(prices)
    weights = (
        pd.Series(args.weights, index=args.tickers)
        if args.weights
        else pd.Series(1 / len(args.tickers), index=args.tickers)
    )
    port = Portfolio(rets, weights, args.value)
    pr = port.pnl_returns

    REPORTS.mkdir(exist_ok=True)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=== Portfolio summary ===")
    for k, v in port.summary().items():
        print(f"{k:>20}: {v:,.4f}" if isinstance(v, float) else f"{k:>20}: {v}")

    print(f"\n=== 1-day VaR / ES (currency, portfolio value {args.value:,.0f}) ===")
    vt = var_table(pr, rets, port.weights, args.value)
    print(vt.round(0))

    print("\n=== Risk contribution (parametric 99% VaR, Euler allocation) ===")
    print(port.component_var(0.99).round(4))

    g = fit_garch(pr)
    print("\n=== GARCH(1,1) on portfolio returns (maximum likelihood) ===")
    print(f"omega {g.omega:.3e} | alpha {g.alpha:.3f} | beta {g.beta:.3f} | persistence {g.persistence:.3f} | "
          f"shock half-life {g.half_life_days:.0f} days")
    print(f"long-run vol {g.long_run_vol * 252 ** 0.5:.1%} p.a. | forecast for tomorrow "
          f"{g.next_sigma * 252 ** 0.5:.1%} p.a.")
    gj = fit_garch(pr, "gjr")
    print("\n=== GJR-GARCH(1,1): negative shocks add gamma * e^2 to tomorrow's variance ===")
    print(f"alpha {gj.alpha:.3f} | gamma {gj.gamma:.3f} | beta {gj.beta:.3f} | persistence {gj.persistence:.3f}")
    print(f"AIC {gj.aic:,.1f} vs GARCH {g.aic:,.1f} | BIC {gj.bic:,.1f} vs GARCH {g.bic:,.1f} "
          f"| LR = {2 * (gj.loglik - g.loglik):.1f} (5% critical value {chi2.ppf(0.95, 1):.2f})")

    print("\n=== Backtest 99% VaR, same out-of-sample period for all methods ===")
    print("(historical / normal / t: 250-day window; FHS-GARCH / FHS-GJR: 750-day window, refit every 20 days)")
    bt_table = backtest_report(pr, 0.99, 250)
    print(bt_table)

    print("\n=== Expected Shortfall backtest (97.5%, Acerbi-Szekely Z2; reject if Z2 < -0.7) ===")
    fhs975 = rolling_fhs(pr, 0.975, 750)
    hs975 = rolling_var(pr, 0.975, 250, "historical").loc[fhs975.index[0]:]
    hs_es = pr.rolling(250).apply(lambda x: -x[x <= np.quantile(x, 0.025)].mean(), raw=True).shift(1)
    z_hs = es_backtest_z2(hs975["return"], hs975["var"], hs_es.loc[hs975.index], 0.975)
    z_fhs = es_backtest_z2(fhs975["return"], fhs975["var"], fhs975["es"], 0.975)
    gjr975 = rolling_fhs(pr, 0.975, 750, model="gjr")
    z_gjr = es_backtest_z2(gjr975["return"], gjr975["var"], gjr975["es"], 0.975)
    z_table = pd.DataFrame({"historical": z_hs, "fhs_garch": z_fhs, "fhs_gjr": z_gjr}).T
    print(z_table)

    print("\n=== Historical stress scenarios ===")
    hs = run_historical_stress(rets, port.weights, args.value)
    print(hs if not hs.empty else "(no scenario windows covered by the data)")

    driver = args.tickers[0]
    print(f"\n=== Hypothetical shock: {driver} -20% (others via regression betas) ===")
    print(run_hypothetical_shock(rets, port.weights, {driver: -0.20}, args.value).round(4))

    # Charts (palette: "Quiet Analytics")
    navy, steel, stone, warm, charcoal = "#101828", "#5B7C99", "#A3A3A3", "#FAFAF7", "#111111"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": warm, "axes.facecolor": warm, "savefig.facecolor": warm,
        "axes.edgecolor": stone, "axes.labelcolor": charcoal, "text.color": charcoal,
        "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "regular", "axes.titlesize": 13,
        "axes.titlelocation": "left", "grid.color": stone, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })

    fhs = rolling_fhs(pr, 0.99, 750)
    bt = rolling_var(pr, 0.99, 250, "historical").loc[fhs.index[0]:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(bt.index, bt["return"] * 100, lw=0.5, color=stone, label="Daily return")
    ax.plot(bt.index, -bt["var"] * 100, lw=1.3, color=navy,
            label=f"VaR 99%, historical ({int(bt['exception'].sum())} exceptions)")
    ax.plot(fhs.index, -fhs["var"] * 100, lw=1.0, color=steel,
            label=f"VaR 99%, FHS-GARCH ({int(fhs['exception'].sum())} exceptions)")
    exc = bt[bt["exception"]]
    ax.scatter(exc.index, exc["return"] * 100, color=navy, s=10, zorder=3, linewidths=0)
    ax.set_ylabel("Daily return, %")
    ax.set_title("VaR backtest: daily returns against the 1-day 99% VaR forecast", pad=12)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(REPORTS / "var_backtest.png", dpi=150)

    cv = port.component_var(0.99)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [steel if i == 0 else navy for i in range(len(cv))]
    ax.barh(cv.index, cv["pct_of_total"] * 100, color=colors, height=0.6)
    for y, v in enumerate(cv["pct_of_total"] * 100):
        ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=9, color=charcoal)
    ax.set_xlabel("Share of total 99% VaR, %")
    ax.set_title("Risk contribution by asset", pad=12)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(REPORTS / "risk_contribution.png", dpi=150)

    data_note = "SIMULATED prices (demo mode)" if args.demo else f"Yahoo Finance prices from {prices.index[0].date()} to {prices.index[-1].date()}"
    (REPORTS / "summary.md").write_text(
        f"# Risk report\n\nPortfolio: {', '.join(f'{t} {w:.0%}' for t, w in port.weights.items())}\n\nData: {data_note}\n\n"
        f"## 1-day VaR / ES (portfolio value {args.value:,.0f})\n\n```\n{_fmt(vt, {})}\n```\n\n"
        f"## GARCH(1,1)\n\nalpha {g.alpha:.3f}, beta {g.beta:.3f}, persistence {g.persistence:.3f}, "
        f"half-life {g.half_life_days:.0f} days\n\n"
        f"GJR-GARCH: alpha {gj.alpha:.3f}, gamma {gj.gamma:.3f}, beta {gj.beta:.3f}; "
        f"AIC {gj.aic:,.1f} vs {g.aic:,.1f} for GARCH\n\n"
        f"## Backtest, 99% VaR\n\n```\n{_fmt(bt_table, {"expected": "{:.1f}", "kupiec_p": "{:.4f}", "independence_p": "{:.4f}", "cond_coverage_p": "{:.4f}"})}\n```\n\n"
        f"## ES backtest, 97.5% (Z2)\n\n```\n{_fmt(z_table, {"Z2": "{:.3f}", "expected": "{:.1f}"})}\n```\n\n"
        f"## Historical stress\n\n```\n{_fmt(hs, {"portfolio_return": "{:.1%}", "max_drawdown_in_window": "{:.1%}"}) if not hs.empty else 'not covered by data'}\n```\n")
    vt.round(2).to_csv(REPORTS / "var_table.csv")
    bt_table.to_csv(REPORTS / "backtest.csv")
    print(f"\nCharts and tables saved to {REPORTS}/")


if __name__ == "__main__":
    main()
