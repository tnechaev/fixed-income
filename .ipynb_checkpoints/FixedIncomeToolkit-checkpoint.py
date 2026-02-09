# fixed_income_toolkit.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
from scipy.linalg import svd
import sdmx  # required for ECB fetching; if you won't fetch, you can omit or mock fetch methods
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.ar_model import AutoReg

class FixedIncomeToolkit:
    """
    Enhanced FixedIncomeToolkit:
    - Nelson-Siegel yield handling + NS fitting per date
    - Kalman filter for time-varying NS betas
    - PCA of yield changes
    - Simple bootstrap (interpolation) zero-curve builder + OIS placeholder
    - Multi-factor NS shocks (level/slope/curvature)
    - Credit spread shocks
    - CVaR minimization via LP with exact duration constraint
    - Rolling backtest (rolling estimation, monthly rebalance, transaction costs)
    """
    def __init__(self):
        self.bonds = []
        self.scenario_returns = None
        # properties for SDMX fetcher
        self.ecb_client = None
        self.ecb_dataflow_id = "YC"

    # -------------------------
    # SDMX / ECB fetcher + utils
    # -------------------------
    def init_ecb_fetcher(self):
        self.ecb_client = sdmx.Client("ECB")

    def fetch_ecb_yield_curve(self, maturities=None, start_period=None, end_period=None):
        """
        Fetch ECB yield curve via SDMX and attempt to flatten columns to tenor codes like 'SR_2Y'.
        Returns DataFrame indexed by date.
        """
        if self.ecb_client is None:
            raise RuntimeError("Call init_ecb_fetcher() before fetching ECB data.")
        if maturities is None:
            maturities = ["SR_3M","SR_6M","SR_1Y","SR_2Y","SR_3Y","SR_5Y","SR_10Y","SR_20Y","SR_30Y"]
        keys = {"DATA_TYPE_FM": maturities, "INSTRUMENT_FM": ["G_N_A"], "FREQ": "B"}
        msg = self.ecb_client.data(self.ecb_dataflow_id, key=keys,
                                   params={"startPeriod": start_period, "endPeriod": end_period})
        df = sdmx.to_pandas(msg, datetime={"dim": "TIME_PERIOD"})
        df.index = pd.to_datetime(df.index)
        # Flatten MultiIndex columns robustly:
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                found = None
                for lvl in col:
                    if isinstance(lvl, str) and lvl.startswith("SR_"):
                        found = lvl; break
                if found is None:
                    for lvl in col:
                        if isinstance(lvl, str) and (("Y" in lvl and any(ch.isdigit() for ch in lvl)) or ("M" in lvl and any(ch.isdigit() for ch in lvl))):
                            found = lvl if lvl.startswith("SR_") else "SR_" + lvl; break
                if found is None:
                    found = "_".join([str(x) for x in col if x is not None])
                new_cols.append(found)
            df.columns = pd.Index(new_cols)
        else:
            df.columns = df.columns.astype(str)
        return df

    def discount_factors_from_ecb(self, yc_df, desired_maturities=None):
        """
        Robust conversion from fetched DataFrame to tidy Date/T/ZeroRate/DF DataFrame.
        desired_maturities example: ["SR_2Y","SR_5Y","SR_10Y"]
        """
        yc = yc_df.copy()
        yc.columns = yc.columns.astype(str)
        if desired_maturities is None:
            desired_maturities = list(yc.columns)
        # build mapping from requested tenor to actual column
        col_map = {}
        available = list(yc.columns)
        for m in desired_maturities:
            if m in yc.columns:
                col_map[m] = m; continue
            token = m.replace("SR_","")
            candidates = [c for c in available if token in c]
            if len(candidates) == 1:
                col_map[m] = candidates[0]; continue
            if len(candidates) > 1:
                sr_cands = [c for c in candidates if c.startswith("SR_")]
                col_map[m] = sr_cands[0] if sr_cands else candidates[0]; continue
            # looser fallback:
            num = ''.join([ch for ch in token if ch.isdigit()])
            fallback = [c for c in available if (num in c and ('Y' in c or 'M' in c))]
            col_map[m] = fallback[0] if fallback else None
        missing = [k for k,v in col_map.items() if v is None]
        if missing:
            raise KeyError(f"Requested maturities not found: {missing}. Available sample: {available[:20]}")
        selected = [col_map[m] for m in desired_maturities]
        df_sel = yc[selected].dropna(how="all").copy()
        rename_map = {col_map[m]: m for m in desired_maturities}
        df_sel = df_sel.rename(columns=rename_map)
        def mat_to_years(s):
            s = s.replace("SR_","")
            if "M" in s: return float(s.replace("M",""))/12.0
            if "Y" in s: return float(s.replace("Y",""))
            try: return float(s)
            except: return None
        tenors = np.array([mat_to_years(m) for m in desired_maturities])
        out_rows = []
        for date, row in df_sel.iterrows():
            rates = row.values.astype(float) / 100.0
            dfs = np.exp(-rates * tenors)
            df_row = pd.DataFrame({"Date":[date]*len(tenors), "T":tenors, "ZeroRate":rates, "DF":dfs})
            out_rows.append(df_row)
        return pd.concat(out_rows, ignore_index=True)

    # -------------------------
    # NS model / pricing / analytics
    # -------------------------
    @staticmethod
    def nelson_siegel_yield(beta0, beta1, beta2, tau, maturities):
        t = np.array(maturities, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = (1 - np.exp(-t / tau)) / (t / tau)
            term = np.where(t==0, 1.0, term)
        return beta0 + beta1 * term + beta2 * (term - np.exp(-t/tau))

    @staticmethod
    def price_fixed_rate_bond(face, coupon_rate, maturity, yield_curve_func, freq=1):
        n_periods = int(np.round(maturity * freq))
        times = np.linspace(1/freq, n_periods/freq, n_periods)
        coupon = coupon_rate * face / freq
        cashflows = np.full(n_periods, coupon); cashflows[-1] += face
        yields = yield_curve_func(times)
        dfs = np.exp(-yields * times)
        price = np.sum(cashflows * dfs)
        return price, times, cashflows, yields, dfs

    @staticmethod
    def macaulay_duration(price, times, cashflows, yields):
        dfs = np.exp(-yields * times)
        pv = cashflows * dfs
        w = pv / np.sum(pv)
        return np.sum(w * times)

    @staticmethod
    def modified_duration(macaulay_duration, yields_mean):
        return macaulay_duration

    @staticmethod
    def convexity(price, times, cashflows, yields):
        dfs = np.exp(-yields * times)
        pv = cashflows * dfs
        return np.sum((times**2) * pv) / np.sum(pv)

    def add_bond(self, name, face, coupon_rate, maturity, freq=1, ns_params=None):
        if ns_params is None: ns_params = (0.02,0.0,0.0,1.0)
        self.bonds.append({"name":name,"face":face,"coupon_rate":coupon_rate,"maturity":maturity,"freq":freq,"ns_params":ns_params})

    def price_all_bonds(self, ns_params=None):
        rows=[]
        for b in self.bonds:
            params = ns_params if ns_params is not None else b["ns_params"]
            beta0,beta1,beta2,tau = params
            yc = lambda tt: FixedIncomeToolkit.nelson_siegel_yield(beta0,beta1,beta2,tau,tt)
            price,times,cashflows,yields,dfs = FixedIncomeToolkit.price_fixed_rate_bond(b["face"], b["coupon_rate"], b["maturity"], yc, b["freq"])
            mac = FixedIncomeToolkit.macaulay_duration(price,times,cashflows,yields)
            mod = FixedIncomeToolkit.modified_duration(mac, np.mean(yields))
            conv = FixedIncomeToolkit.convexity(price,times,cashflows,yields)
            rows.append({"name":b["name"],"price":price,"macaulay_duration":mac,"modified_duration":mod,"convexity":conv,"maturity":b["maturity"],"coupon":b["coupon_rate"]})
        return pd.DataFrame(rows)

    # -------------------------
    # Multi-factor scenario generation
    # -------------------------
    def simulate_ns_factor_shocks(self, n_scenarios=2000, level_vol=0.008, slope_vol=0.004, curvature_vol=0.003, seed=42):
        rng = np.random.default_rng(seed)
        shocks = rng.normal(loc=0.0, scale=[level_vol, slope_vol, curvature_vol], size=(n_scenarios,3))
        self.ns_factor_shocks = shocks
        return shocks

    def generate_scenario_returns_multi_factor(self, horizon_years=1/12, n_scenarios=2000, level_vol=0.008, slope_vol=0.004, curvature_vol=0.003, seed=42, include_spread=False, spread_vol=0.001):
        shocks = self.simulate_ns_factor_shocks(n_scenarios, level_vol, slope_vol, curvature_vol, seed)
        scenario_returns = np.zeros((n_scenarios, len(self.bonds)))
        for i, (dl,ds,dc) in enumerate(shocks):
            for j,b in enumerate(self.bonds):
                beta0,beta1,beta2,tau = b["ns_params"]
                beta0_s = beta0 + dl; beta1_s = beta1 + ds; beta2_s = beta2 + dc
                yc = lambda t: FixedIncomeToolkit.nelson_siegel_yield(beta0_s,beta1_s,beta2_s,tau,t)
                price, times, cashflows, yields, dfs = FixedIncomeToolkit.price_fixed_rate_bond(b["face"],b["coupon_rate"],b["maturity"],yc,b["freq"])
                coupon = b["coupon_rate"] * b["face"]
                coupon_received = coupon * horizon_years
                # optional credit spread shock: add on top as an additional yield shift
                if include_spread:
                    spread_shift = np.random.default_rng(seed+i+j).normal(0, spread_vol)
                    price = price * np.exp(-spread_shift * b["maturity"])  # fiat simple adjustment
                total_return = (price + coupon_received - b["face"]) / b["face"]
                scenario_returns[i,j] = total_return
        self.scenario_returns = scenario_returns
        return scenario_returns

    # -------------------------
    # CVaR LP with exact duration constraints
    # -------------------------
    def minimize_cvar(self, alpha=0.95, target_return=None, bounds=None, target_duration=None, duration_tol=None):
        """
        LP CVaR: adds optional exact duration equality or duration band (inequalities).
        Returns (result_dict, linprog_res, extras)
        """
        R = self.scenario_returns
        if R is None:
            raise ValueError("No scenario returns generated.")
        T,N = R.shape
        num_vars = N + 1 + T
        c = np.zeros(num_vars)
        c[N] = 1.0
        c[N+1:] = 1.0 / ((1-alpha)*T)

        # primary A_ub for z >= -R_i w - eta  ->  -R_i w - eta - z_i <= 0
        A_ub = np.zeros((2*T, num_vars)); b_ub = np.zeros(2*T)
        for i in range(T):
            A_ub[i, :N] = -R[i]
            A_ub[i, N] = -1.0
            A_ub[i, N+1+i] = -1.0
        for i in range(T):
            A_ub[T+i, N+1+i] = -1.0

        # equality sum w =1
        A_eq = np.zeros((1, num_vars)); A_eq[0,:N] = 1.0
        b_eq = np.array([1.0])

        # add return constraint if needed as inequality (E[R]w >= target_return -> -E[R]w <= -target_return)
        if target_return is not None:
            ER = np.mean(R, axis=0)
            row = np.zeros(num_vars); row[:N] = -ER
            A_ub = np.vstack([A_ub, row]); b_ub = np.append(b_ub, -target_return)

        # Add duration exact equality or band
        if target_duration is not None:
            df_prices = self.price_all_bonds()
            durations = df_prices["modified_duration"].values
            if duration_tol is None:
                # equality
                A_eq = np.vstack([A_eq, np.zeros((1, num_vars))])
                A_eq[-1, :N] = durations
                b_eq = np.append(b_eq, target_duration)
            else:
                row_up = np.zeros(num_vars); row_up[:N] = durations
                A_ub = np.vstack([A_ub, row_up]); b_ub = np.append(b_ub, target_duration + duration_tol)
                row_lo = np.zeros(num_vars); row_lo[:N] = -durations
                A_ub = np.vstack([A_ub, row_lo]); b_ub = np.append(b_ub, -(target_duration - duration_tol))

        # variable bounds
        if bounds is None:
            var_bounds = [(0.0,1.0)]*N + [(None,None)] + [(0.0,None)]*T
        else:
            var_bounds = list(bounds) + [(None,None)] + [(0.0,None)]*T

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=var_bounds, method='highs')
        if not res.success:
            return None, res, None
        w = res.x[:N]; eta = res.x[N]; z = res.x[N+1:]
        cvar = eta + np.sum(z) / ((1-alpha)*T)
        port_returns = self.scenario_returns @ w
        return {"weights":w, "eta":eta, "cvar":cvar, "mean_return":np.mean(port_returns), "vol":np.std(port_returns)}, res, {"port_returns":port_returns}

    # -------------------------
    # NS fitting & PCA
    # -------------------------
    @staticmethod
    def fit_ns_for_date(maturities, yields):
        maturities = np.array(maturities, dtype=float)
        yields = np.array(yields, dtype=float)
        def model(t, b0,b1,b2,tau):
            term = (1 - np.exp(-t/tau)) / (t/tau)
            return b0 + b1*term + b2*(term - np.exp(-t/tau))
        def loss(params):
            return np.sum((model(maturities, *params) - yields)**2)
        init = [yields.mean(), 0.0, 0.0, 1.0]
        res = minimize(loss, init, bounds=[(None,None),(None,None),(None,None),(0.01,5.0)])
        return tuple(res.x) if res.success else tuple(init)

    def fit_ns_time_series(self, yield_df, maturities_years):
        records=[]
        for date, row in yield_df.iterrows():
            yields = row.values.astype(float)
            params = FixedIncomeToolkit.fit_ns_for_date(maturities_years, yields)
            records.append((date,)+params)
        return pd.DataFrame(records, columns=["Date","beta0","beta1","beta2","tau"]).set_index("Date")

    def pca_yield_changes(self, yield_df, n_components=3):
        dy = yield_df.diff().dropna()
        X = dy.values - np.mean(dy.values, axis=0)
        U,S,Vt = svd(X, full_matrices=False)
        components = Vt[:n_components,:]
        explained = (S**2) / np.sum(S**2)
        scores = X @ components.T
        return {"components":components, "explained":explained[:n_components], "scores":scores, "maturities":yield_df.columns.tolist()}

    # -------------------------
    # Kalman filter for NS betas (simple)
    # -------------------------
    def kalman_filter_ns(self, yield_df, maturities_years, tau_fixed=1.0, R_obs=None, Q_state=None):
        """
        Estimate time-varying beta0,beta1,beta2 using linear KF with state = [b0,b1,b2]' and fixed tau.
        This is a basic implementation; for production, consider statsmodels or pykalman.
        """
        dates = list(yield_df.index)
        m = len(maturities_years)
        t = np.array(maturities_years)
        tau = float(tau_fixed)
        term = (1 - np.exp(-t / tau)) / (t / tau)
        H = np.column_stack([np.ones_like(t), term, term - np.exp(-t/tau)])

        if R_obs is None:
            R_obs = np.eye(m) * 1e-6
        if Q_state is None:
            Q_state = np.eye(3) * 1e-6

        y0 = yield_df.iloc[0].values.astype(float)
        init = self.fit_ns_for_date(maturities_years, y0)
        x = np.array(init[:3])
        P = np.eye(3) * 1.0

        xs = []
        for i in range(len(dates)):
            y = yield_df.iloc[i].values.astype(float)
            # predict (random walk)
            x_prior = x.copy()
            P_prior = P + Q_state
            # update
            S = H @ P_prior @ H.T + R_obs
            K = P_prior @ H.T @ np.linalg.inv(S)
            innovation = y - H @ x_prior
            x = x_prior + K @ innovation
            P = (np.eye(3) - K @ H) @ P_prior
            xs.append(x.copy())
        df_out = pd.DataFrame(xs, index=dates, columns=["beta0_kf","beta1_kf","beta2_kf"])
        return df_out

    # -------------------------
    # Simple bootstrapping (interpolation) and OIS stub
    # -------------------------
    def bootstrap_zero_curve_from_points(self, tenor_points):
        """
        tenor_points: dict {T_years: zero_rate_decimal}
        Returns interpolation function zero_rate(t)
        """
        if isinstance(tenor_points, dict):
            items = sorted(tenor_points.items())
        else:
            items = sorted(tenor_points)
        T = np.array([i[0] for i in items]); y = np.array([i[1] for i in items])
        def zero_rate_func(tq):
            tq = np.array(tq, dtype=float)
            return np.interp(tq, T, y, left=y[0], right=y[-1])
        return zero_rate_func

    # -------------------------
    # Credit spread shocks (simple)
    # -------------------------
    def add_credit_spread_shocks(self, scenarios, spread_vol=0.001, seed=42):
        rng = np.random.default_rng(seed)
        T,N = scenarios.shape
        spread = rng.normal(0, spread_vol, size=(T,N))
        return scenarios + spread

    # -------------------------
    # Rolling backtest: estimation -> optimize -> rebalance
    # -------------------------
    def backtest_rolling(self, yield_series_df, maturities_years, window=252, rebalance_days=21,
                         alpha=0.95, target_duration=None, duration_tol=None, tc=0.0005):
        dates = yield_series_df.index
        steps = max(0, (len(dates) - window) // rebalance_days)
        weights_hist=[]; perf_hist=[]; bench_hist=[]; turnovers=[]
        prev_w = None
        for k in range(steps):
            start = k*rebalance_days
            train = yield_series_df.iloc[start:start+window]
            test = yield_series_df.iloc[start+window:start+window+rebalance_days]
            if len(test)==0: break
            params = self.fit_ns_for_date(maturities_years, train.iloc[-1].values)
            for b in self.bonds: b["ns_params"] = params
            # estimate vol from fitted betas in train
            betas = self.fit_ns_time_series(train, maturities_years)
            if len(betas) > 2:
                vol_est = np.std(betas.diff().dropna().values, axis=0)
            else:
                vol_est = np.array([0.005,0.002,0.001])
            scenarios = self.generate_scenario_returns_multi_factor(horizon_years=rebalance_days/252,
                                                                   n_scenarios=500,
                                                                   level_vol=max(vol_est[0], 1e-4),
                                                                   slope_vol=max(vol_est[1], 1e-4),
                                                                   curvature_vol=max(vol_est[2], 1e-4),
                                                                   seed=123+k)
            # optimize with LP exact duration
            opt,res,extras = self.minimize_cvar(alpha=alpha, target_return=None, bounds=None, target_duration=target_duration, duration_tol=duration_tol)
            if opt is None:
                w = np.ones(len(self.bonds))/len(self.bonds)
            else:
                w = opt["weights"]
            # compute turnover and tc
            turnover = np.sum(np.abs(w - (prev_w if prev_w is not None else np.zeros_like(w))))
            tc_cost = tc * turnover
            realized = np.mean(scenarios, axis=0) @ w - tc_cost
            bench_w = np.ones(len(self.bonds))/len(self.bonds)
            bench_ret = np.mean(scenarios, axis=0) @ bench_w - tc * np.sum(np.abs(bench_w - (prev_w if prev_w is not None else np.zeros_like(bench_w))))
            weights_hist.append(w); perf_hist.append(realized); bench_hist.append(bench_ret); turnovers.append(turnover)
            prev_w = w.copy()
        return {"strategy": np.array(perf_hist), "benchmark":np.array(bench_hist), "weights":np.array(weights_hist), "turnovers":np.array(turnovers)}


    class NelsonSiegelDNS(MLEModel):
        def __init__(self, yields, maturities, tau=2.5):
            self.maturities = np.array(maturities)
            self.tau = tau
            k_obs = yields.shape[1]
            super().__init__(
                endog=yields,
                k_states=3,
                k_posdef=3,
                initialization="approximate_diffuse"
            )
            # Transition = Identity → simple random walk
            self['transition'] = np.eye(3)
            self['selection']  = np.eye(3)
            self['state_cov']  = np.eye(3) * 1e-4  # small state noise
            self['obs_cov']    = np.eye(k_obs) * 1e-3  # measurement noise

        def update(self, params, **kwargs):
            t = self.maturities
            tau = self.tau
            term = (1 - np.exp(-t/tau)) / (t/tau)
            H = np.column_stack([
                np.ones_like(t),          # level
                term,                     # slope
                term - np.exp(-t/tau)     # curvature
            ])
            self['design'] = H

        @property
        def start_params(self):
            return np.zeros(3)  # initial beta0, beta1, beta2

    # -------------------------
    # Fit dynamic NS and return smoothed states
    # -------------------------
    def fit_dynamic_ns(self, yield_df, tau=2.5):
        """
        Fit Dynamic Nelson-Siegel using Kalman filter.
        yield_df: DataFrame indexed by date, columns = maturities in years, yields in decimals
        """
        maturities = yield_df.columns.values.astype(float)
        model = self.NelsonSiegelDNS(yield_df.values, maturities=maturities, tau=tau)
        res = model.fit(disp=False)
        smoothed_states = res.smoothed_state.T  # T x 3 (level, slope, curvature)
        return smoothed_states, res

    def forecast_yield_curve_ar1(self, smoothed_states, yield_df, days_ahead=10, tau=2.5):
        """
        Forecast multiple days ahead using AR(1) on smoothed DNS factors.
        smoothed_states: T x 3 array (level, slope, curvature)
        days_ahead: number of future days to forecast
        yield_df: original yield DataFrame (for maturities)
        tau: NS decay parameter
        Returns: forecasted yields array (days_ahead x maturities)
        """
        factors_forecast = np.zeros((days_ahead, 3))
        for i in range(3):  # for each factor: level, slope, curvature
            series = smoothed_states[:, i]
            ar_model = AutoReg(series, lags=1, old_names=False).fit()
            factors_forecast[:, i] = ar_model.predict(start=len(series), end=len(series)+days_ahead-1)

        maturities = yield_df.columns.values.astype(float)
        term = (1 - np.exp(-maturities / tau)) / (maturities / tau)
        H = np.column_stack([
            np.ones_like(maturities),
            term,
            term - np.exp(-maturities / tau)
        ])

        # Compute yield curve for each forecasted day
        forecast_curves = np.array([H @ factors_forecast[i, :] for i in range(days_ahead)])
        return forecast_curves
