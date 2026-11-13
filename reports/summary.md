# Risk report

Portfolio: SPY 35%, TLT 25%, GLD 10%, EFA 15%, QQQ 15%

Data: Yahoo Finance prices from 2005-01-03 to 2026-11-13

## 1-day VaR / ES (portfolio value 1,000,000)

```
         Historical Parametric (normal) Parametric (Student-t) Monte Carlo (t, df=5) FHS-GARCH (conditional)
VaR 95%      11,407              12,425                 10,671                11,641                  17,959
VaR 99%      21,281              17,737                 20,746                19,902                  29,284
ES 97.5%     23,135              17,826                 22,876                20,804                  30,238
```

## GARCH(1,1)

alpha 0.106, beta 0.876, persistence 0.983, half-life 40 days

GJR-GARCH: alpha 0.006, gamma 0.141, beta 0.904; AIC 10,495.9 vs 10,646.3 for GARCH

## Backtest, 99% VaR

```
           exceptions expected kupiec_p independence_p cond_coverage_p last_250_exceptions basel_zone
historical         71     47.5   0.0014         0.0001          0.0000                   2      green
normal            104     47.5   0.0000         0.0000          0.0000                   5     yellow
t                  74     47.5   0.0004         0.0000          0.0000                   2      green
fhs_garch          59     47.5   0.1064         0.0065          0.0067                   6     yellow
fhs_gjr            63     47.5   0.0315         0.0616          0.0172                   5     yellow
```

## ES backtest, 97.5% (Z2)

```
                Z2 exceptions expected  reject_5pct
historical  -0.427        155    118.8        False
fhs_garch   -0.108        129    118.8        False
fhs_gjr     -0.101        127    118.8        False
```

## Historical stress

```
                                     days portfolio_return       pnl max_drawdown_in_window
scenario                                                                                   
GFC: Lehman collapse (Sep-Nov 2008)    58           -27.3%  -273,456                 -26.8%
Euro sovereign crisis (Jul-Sep 2011)   64            -1.6%   -16,013                  -8.0%
China devaluation (Aug 2015)           16            -3.7%   -37,465                  -6.9%
Volmageddon (Feb 2018)                 10            -7.0%   -70,056                  -7.6%
COVID crash (Feb-Mar 2020)             24           -18.5%  -185,428                 -19.7%
2022 rates shock (Jan-Oct 2022)       196           -25.9%  -258,517                 -25.6%
SVB / regional banks (Mar 2023)         8             1.5%    15,100                  -0.9%
```
