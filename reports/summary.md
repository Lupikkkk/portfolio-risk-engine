# Risk report

Portfolio: SPY 35%, TLT 25%, GLD 10%, EFA 15%, QQQ 15%

Data: Yahoo Finance prices from 2007-01-03 to 2026-11-13

## 1-day VaR / ES (portfolio value 1,000,000)

```
          Historical  Parametric (normal)  Parametric (Student-t)  Monte Carlo (t, df=5)  FHS-GARCH (conditional)
VaR 95%  11,914.0000          12,796.0000             10,959.0000            11,992.0000              18,470.0000
VaR 99%  21,560.0000          18,260.0000             21,508.0000            20,554.0000              30,448.0000
ES 97.5% 23,923.0000          18,352.0000             23,818.0000            21,415.0000              31,285.0000
```

## GARCH(1,1)

alpha 0.112, beta 0.872, persistence 0.984, half-life 43 days

## Backtest, 99% VaR

```
           exceptions expected kupiec_p independence_p cond_coverage_p last_250_exceptions basel_zone
historical         58  42.5000   0.0234         0.0001          0.0000                   2      green
normal             88  42.5000   0.0000         0.0000          0.0000                   5     yellow
t                  62  42.5000   0.0048         0.0000          0.0000                   2      green
fhs_garch          54  42.5000   0.0882         0.0050          0.0045                   6     yellow
```

## ES backtest, 97.5% (Z2)

```
                Z2 exceptions expected reject_5pct
historical -0.3464        131 106.2000       False
fhs_garch  -0.0564        108 106.2000       False
```

## Historical stress

```
                                      days  portfolio_return           pnl  max_drawdown_in_window
scenario                                                                                          
GFC: Lehman collapse (Sep-Nov 2008)     58           -0.2735 -273,456.7417                 -0.2683
Euro sovereign crisis (Jul-Sep 2011)    64           -0.0160  -16,012.9461                 -0.0795
China devaluation (Aug 2015)            16           -0.0375  -37,464.5671                 -0.0691
Volmageddon (Feb 2018)                  10           -0.0701  -70,055.9356                 -0.0764
COVID crash (Feb-Mar 2020)              24           -0.1854 -185,427.9365                 -0.1973
2022 rates shock (Jan-Oct 2022)        196           -0.2585 -258,516.8527                 -0.2558
SVB / regional banks (Mar 2023)          8            0.0151   15,100.2179                 -0.0086
```
