# Risk report

Portfolio: SPY 35%, TLT 25%, GLD 10%, EFA 15%, QQQ 15%

Data: Yahoo Finance prices from 2005-01-03 to 2026-11-12

## 1-day VaR / ES (portfolio value 1,000,000)

```
         Historical Parametric (normal) Parametric (Student-t) Monte Carlo (t, df=5) FHS-GARCH (conditional)
VaR 95%      11,409              12,425                 10,670                11,640                  18,933
VaR 99%      21,281              17,737                 20,749                19,903                  30,862
ES 97.5%     23,135              17,826                 22,881                20,804                  31,857
```

FHS-GARCH 99% VaR is +45% relative to historical VaR.

## Risk contribution (parametric 99% VaR, Euler)

```
       weight marginal_var component_var pct_of_total
Ticker                                               
SPY       35%       25,528         8,935        49.3%
QQQ       15%       28,229         4,234        23.4%
EFA       15%       26,891         4,034        22.2%
GLD       10%        7,898           790         4.4%
TLT       25%          557           139         0.8%
```

## GARCH(1,1)

alpha 0.107, beta 0.876, persistence 0.983, half-life 40 days

GJR-GARCH: alpha 0.006, gamma 0.141, beta 0.904; AIC 10,493.7 vs 10,644.0 for GARCH; LR = 152.3 (5% critical value 3.84)

## Backtest, 99% VaR

Out-of-sample period: 2007-12-27 to 2026-11-12, 4,750 days. All exception dates: [exceptions.csv](exceptions.csv).

```
           exceptions expected exceptions_2008 kupiec_p independence_p cond_coverage_p last_250_exceptions basel_zone
historical         71     47.5              13   0.0014         0.0001          0.0000                   2      green
normal            104     47.5              16   0.0000         0.0000          0.0000                   5     yellow
t                  74     47.5              12   0.0004         0.0000          0.0000                   2      green
fhs_garch          59     47.5               4   0.1061         0.0065          0.0066                   6     yellow
fhs_gjr            63     47.5               6   0.0313         0.0616          0.0172                   5     yellow
```

### Last 250 days (2025-11-14 to 2026-11-12)

Exception dates:

```
historical   2: 2026-06-05, 2026-11-05
normal       5: 2026-03-20, 2026-05-15, 2026-06-05, 2026-06-10, 2026-11-05
t            2: 2026-06-05, 2026-11-05
fhs_garch    6: 2026-01-30, 2026-03-03, 2026-03-20, 2026-06-05, 2026-10-30, 2026-11-05
fhs_gjr      5: 2026-01-30, 2026-03-03, 2026-05-15, 2026-06-05, 2026-11-05
```

Largest loss in the 250-day window behind historical VaR on 2025-11-14: -3.93% on 2025-04-04; it leaves the window after 2026-04-06.

## ES backtest, 97.5% (Z2)

```
                Z2 exceptions expected  reject_5pct
historical  -0.428        155    118.8        False
fhs_garch   -0.108        129    118.8        False
fhs_gjr     -0.101        127    118.8        False
```

## Current volatility

Realised volatility (annualised): last 20 days 17.0%, last 250 days 12.3%. GARCH forecast for tomorrow 18.1%, GJR-GARCH 18.2%.

Daily moves of 1.5% or more in the last 20 days: 2026-10-30 -1.52%, 2026-11-04 +1.68%, 2026-11-05 -2.81%, 2026-11-09 +2.01%.

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
