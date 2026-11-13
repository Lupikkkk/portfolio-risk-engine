"""Interactive dashboard: `streamlit run app.py`"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk_engine import Portfolio, load_prices, simulate_prices, to_returns
from risk_engine.backtest import backtest_report, rolling_var
from risk_engine.garch import fit_garch, rolling_fhs
from risk_engine.stress import run_historical_stress, run_hypothetical_shock
from risk_engine.var import var_table

st.set_page_config(page_title="Portfolio Risk Engine", layout="wide")
st.title("Portfolio Risk Engine")
st.caption("VaR / Expected Shortfall · backtesting · stress testing · risk attribution")

with st.sidebar:
    tickers = st.text_input("Tickers (space-separated)", "SPY TLT GLD EFA QQQ").split()
    raw_w = st.text_input("Weights (blank = equal)", "0.35 0.25 0.10 0.15 0.15")
    value = st.number_input("Portfolio value", value=1_000_000, step=100_000)
    start = st.date_input("Start date", pd.Timestamp("2007-01-01"))
    demo = st.checkbox("Use simulated data (offline)", value=False)
    alpha = st.select_slider("Confidence", options=[0.95, 0.975, 0.99], value=0.99)


@st.cache_data(show_spinner="Loading prices...")
def get_prices(tickers, start, demo):
    return simulate_prices(list(tickers), 2500) if demo else load_prices(list(tickers), start=str(start))


prices = get_prices(tuple(tickers), start, demo)
rets = to_returns(prices)
weights = pd.Series([float(x) for x in raw_w.split()], index=tickers) if raw_w.strip() else \
    pd.Series(1 / len(tickers), index=tickers)
port = Portfolio(rets, weights, value)
pr = port.pnl_returns

s = port.summary()
c = st.columns(4)
c[0].metric("Annual return", f"{s['annual_return']:.1%}")
c[1].metric("Annual volatility", f"{s['annual_volatility']:.1%}")
c[2].metric("Sharpe (rf=0)", f"{s['sharpe_rf0']:.2f}")
c[3].metric("Max drawdown", f"{s['max_drawdown']:.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["VaR & ES", "Backtest", "Stress tests", "Risk attribution"])

with tab1:
    st.dataframe(var_table(pr, rets, port.weights, value).style.format("{:,.0f}"))
    fig = go.Figure(go.Histogram(x=pr * 100, nbinsx=120, marker_color="#4a5568"))
    v = np.quantile(pr, 1 - alpha) * 100
    fig.add_vline(x=v, line_color="#c53030", annotation_text=f"Hist. VaR {alpha:.1%}")
    fig.update_layout(title="Distribution of daily portfolio returns (%)", height=380)
    st.plotly_chart(fig, width="stretch")

with tab2:
    method = st.radio("Method", ["historical", "normal", "t", "fhs_garch", "fhs_gjr"], horizontal=True)
    bt = (rolling_fhs(pr, alpha, 750, model="gjr" if method == "fhs_gjr" else "garch")
          if method.startswith("fhs") else rolling_var(pr, alpha, 250, method))
    g = fit_garch(pr)
    st.caption(f"GARCH(1,1): alpha {g.alpha:.3f}, beta {g.beta:.3f}, persistence {g.persistence:.3f}, "
               f"vol forecast {g.next_sigma * 252 ** 0.5:.1%} p.a. (long-run {g.long_run_vol * 252 ** 0.5:.1%})")
    fig = go.Figure()
    fig.add_scatter(x=bt.index, y=bt["return"] * 100, mode="lines", name="Return", line=dict(width=0.7))
    fig.add_scatter(x=bt.index, y=-bt["var"] * 100, mode="lines", name="-VaR", line=dict(color="#c53030"))
    exc = bt[bt["exception"]]
    fig.add_scatter(x=exc.index, y=exc["return"] * 100, mode="markers", name="Exceptions",
                    marker=dict(color="#c53030", size=5))
    fig.update_layout(height=420, yaxis_title="%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(backtest_report(pr, alpha, 250))

with tab3:
    st.subheader("Historical scenarios")
    hs = run_historical_stress(rets, port.weights, value)
    st.dataframe(hs.style.format({"portfolio_return": "{:.2%}", "pnl": "{:,.0f}", "max_drawdown_in_window": "{:.2%}"})
                 if not hs.empty else hs)
    st.subheader("Hypothetical shock")
    drv = st.selectbox("Driver asset", tickers)
    shock = st.slider("Shock to driver (%)", -50, 50, -20) / 100
    st.dataframe(run_hypothetical_shock(rets, port.weights, {drv: shock}, value).style.format("{:,.4f}"))

with tab4:
    cv = port.component_var(alpha)
    st.dataframe(cv.style.format({"weight": "{:.1%}", "marginal_var": "{:,.0f}", "component_var": "{:,.0f}",
                                  "pct_of_total": "{:.1%}"}))
    fig = go.Figure(go.Bar(x=cv["pct_of_total"] * 100, y=cv.index, orientation="h", marker_color="#2b6cb0"))
    fig.update_layout(title="% of total VaR", height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
