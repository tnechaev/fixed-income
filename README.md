# Fixed Income Toolkit

This is an ongoing quant research project for **fixed-income analytics, yield-curve modeling, and bond portfolio simulation/optimization**.  
It provides tools to fetch ECB yield data, fit Nelson-Siegel (NS) and Svensson (NSS) curves, estimate time-varying factors with an EKF+RTS smoother, run VAR forecasts, simulate multi-factor scenario returns, and construct CVaR-optimized bond portfolios.

---

## Attention!

This project is a **research-grade demo, not production**. Limitations include:

- ECB SDMX endpoints and field names can change — the fetcher may need maintenance.
- Curve fitting and EKF hyperparameters are heuristic and dataset-dependent.
- Bond pricing uses simplified day-count assumptions and continuous compounding.
- Scenario generation is model-based; it does not replace market-calibrated risk models.
- CVaR optimizer depends on scenario quality.

---

## Project Objectives

- Provide a way for parametric yield modeling (NS / NSS).
- Fetch and preprocess historical Euro area yield curves from ECB.
- Fit per-date parametric curves and produce **time series of factors**.
- Estimate **dynamic factors** (level, slope, curvature, and tau) using EKF + RTS smoothing.
- Forecast factor evolution using VAR and produce stochastic yield scenarios.
- Simulate bond returns under multi-factor NS shocks (and optional credit spread shocks).
- Construct portfolios with **CVaR minimization**, optionally with duration constraints.
- Provide a rolling backtest wrapper to prototype strategy performance.

---

## Model Overview

### Nelson-Siegel (NS) & Svensson (NSS)

- NS models yields with 3 factors plus a decay parameter (tau).
- NSS adds a fourth factor and second decay parameter to capture more curve shapes.
- Both models allow reconstructing yield curves from factor states.

### EKF + RTS Smoother

- Extended Kalman Filter (EKF) estimates dynamic factor states from historical yields.
- RTS smoother refines factor estimates using forward-backward smoothing.
- Supports both NS (4-state) and NSS (6-state) models.

### VAR Forecasting & Stochastic Scenarios

- Fits a VAR model to smoothed factor states.
- Simulates forward factor scenarios and reconstructs yield curves for each scenario.
- Produces scenario clouds and mean forecasts for portfolio simulations.

### Bond Analytics & Scenario Returns

- Price fixed-rate bonds using discounting of coupons under parametric yield curves.
- Compute Macaulay duration, modified duration, and convexity.
- Generate scenario-based bond returns using factor shocks.
- Optimize portfolio weights via CVaR minimization with optional constraints.

### Backtesting

- Rolling backtest framework using historical yields.
- Rebalances portfolios at configurable intervals.
- Supports transaction costs and evaluates realized strategy performance.
