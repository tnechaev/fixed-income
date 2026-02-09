# fixed_income_toolkit.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
from scipy.linalg import svd
import sdmx
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg

class FixedIncomeToolkit:
    """
    Features:
      - ECB SDMX fetching helpers
      - Nelson-Siegel (NS) and Svensson (NSS) yield helpers
      - Fit NS per date and NSS linear solve given taus
      - Extended Kalman Filter (EKF) + RTS smoother for dynamic NS/NSS (log-tau state)
      - VAR forecasting of smoothed states (supports both NS and NSS)
      - PCA on yield changes
      - Simple bootstrap zero curve interpolation
      - Bond registry, pricing and analytics (NS pricing)
      - Scenario generation (NS factor shocks) and CVaR LP optimizer
      - Rolling backtest wrapper
    """
    def __init__(self):
        self.bonds = []
        self.scenario_returns = None
        self.ecb_client = None
        self.ecb_dataflow_id = "YC"

    # -------------------------
    # SDMX / ECB fetcher
    # -------------------------
    def init_ecb_fetcher(self):
        self.ecb_client = sdmx.Client("ECB")

    def fetch_ecb_yield_curve(self, maturities=None, start_period=None, end_period=None):
        if self.ecb_client is None:
            raise RuntimeError("Call init_ecb_fetcher() before fetching ECB data.")
        if maturities is None:
            maturities = ["SR_3M","SR_6M","SR_1Y","SR_2Y","SR_3Y","SR_5Y","SR_7Y","SR_10Y","SR_15Y","SR_20Y","SR_30Y"]
        keys = {"DATA_TYPE_FM": maturities, "INSTRUMENT_FM": ["G_N_A"], "FREQ": "B"}
        msg = self.ecb_client.data(self.ecb_dataflow_id, key=keys,
                                   params={"startPeriod": start_period, "endPeriod": end_period})
        df = sdmx.to_pandas(msg, datetime={"dim":"TIME_PERIOD"})
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
        yc = yc_df.copy()
        yc.columns = yc.columns.astype(str)
        if desired_maturities is None:
            desired_maturities = list(yc.columns)
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
    # Nelson-Siegel / Svensson helpers (upgraded)
    # -------------------------
    @staticmethod
    def nelson_siegel_yield(beta0, beta1, beta2, tau, maturities):
        t = np.array(maturities, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = (1 - np.exp(-t / tau)) / (t / tau)
            term = np.where(t == 0, 1.0, term)
        return beta0 + beta1 * term + beta2 * (term - np.exp(-t / tau))

    @staticmethod
    def svensson_yield(beta0, beta1, beta2, beta3, tau1, tau2, maturities):
        t = np.array(maturities, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            x1 = t / tau1
            exp1 = np.exp(-x1)
            A1 = (1 - exp1) / x1
            A1 = np.where(x1 == 0, 1.0, A1)
            B1 = A1 - exp1

            x2 = t / tau2
            exp2 = np.exp(-x2)
            A2 = (1 - exp2) / x2
            A2 = np.where(x2 == 0, 1.0, A2)
            B2 = A2 - exp2

        return beta0 + beta1 * A1 + beta2 * B1 + beta3 * B2

    @staticmethod
    def ns_yields_from_state(state, maturities, model='ns'):
        """
        state for model='ns': [b0,b1,b2, log_tau]
        state for model='nss': [b0,b1,b2,b3, log_tau1, log_tau2]
        Returns yields vector (N,)
        """
        t = np.array(maturities, dtype=float)
        if model == 'ns':
            if len(state) != 4:
                raise ValueError("NS state must be length 4 (b0,b1,b2,log_tau)")
            b0, b1, b2, log_tau = state
            tau = float(np.exp(log_tau))
            return FixedIncomeToolkit.nelson_siegel_yield(b0, b1, b2, tau, t)

        elif model == 'nss':
            if len(state) != 6:
                raise ValueError("NSS state must be length 6 (b0,b1,b2,b3,log_tau1,log_tau2)")
            b0, b1, b2, b3, log_tau1, log_tau2 = state
            tau1 = float(np.exp(log_tau1))
            tau2 = float(np.exp(log_tau2))
            return FixedIncomeToolkit.svensson_yield(b0, b1, b2, b3, tau1, tau2, t)

        else:
            raise ValueError("model must be 'ns' or 'nss'")

    @staticmethod
    def ns_jacobian_wrt_state(state, maturities, model='ns'):
        """
        Analytic Jacobian of yields wrt state variables.
        For 'ns' returns N x 4 (d/d[b0,b1,b2,log_tau])
        For 'nss' returns N x 6 (d/d[b0,b1,b2,b3,log_tau1,log_tau2])
        """
        t = np.array(maturities, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            if model == 'ns':
                b0, b1, b2, log_tau = state
                tau = float(np.exp(log_tau))
                x = t / tau
                expx = np.exp(-x)
                A = (1 - expx) / x
                A = np.where(x == 0, 1.0, A)
                B = A - expx

                # dA/dx
                denom = np.where(x == 0, 1.0, x**2)
                dA_dx = ((x + 1.0) * expx - 1.0) / denom
                dA_dx = np.where(x == 0, -0.5, dA_dx)  # small-x limit
                # dx/dtau = -t/tau^2 = -x/tau
                dA_dtau = dA_dx * (-x / tau)

                # dexp/dtau = exp(-x) * x / tau
                dexp_dtau = expx * (x / tau)
                dB_dtau = dA_dtau - dexp_dtau

                # dy/dtau = b1 * dA_dtau + b2 * dB_dtau
                dy_dtau = b1 * dA_dtau + b2 * dB_dtau
                d_dlogtau = tau * dy_dtau

                d_db0 = np.ones_like(t)
                d_db1 = A
                d_db2 = B

                H = np.column_stack([d_db0, d_db1, d_db2, d_dlogtau])
                return H

            elif model == 'nss':
                b0, b1, b2, b3, log_tau1, log_tau2 = state
                tau1 = float(np.exp(log_tau1))
                tau2 = float(np.exp(log_tau2))
                x1 = t / tau1
                x2 = t / tau2
                exp1 = np.exp(-x1)
                exp2 = np.exp(-x2)

                A1 = (1 - exp1) / x1
                A1 = np.where(x1 == 0, 1.0, A1)
                B1 = A1 - exp1

                A2 = (1 - exp2) / x2
                A2 = np.where(x2 == 0, 1.0, A2)
                B2 = A2 - exp2

                # derivatives dA1/dtau1 etc
                denom1 = np.where(x1 == 0, 1.0, x1**2)
                dA1_dx1 = ((x1 + 1.0) * exp1 - 1.0) / denom1
                dA1_dx1 = np.where(x1 == 0, -0.5, dA1_dx1)
                dA1_dtau1 = dA1_dx1 * (-x1 / tau1)
                dexp1_dtau1 = exp1 * (x1 / tau1)
                dB1_dtau1 = dA1_dtau1 - dexp1_dtau1

                denom2 = np.where(x2 == 0, 1.0, x2**2)
                dA2_dx2 = ((x2 + 1.0) * exp2 - 1.0) / denom2
                dA2_dx2 = np.where(x2 == 0, -0.5, dA2_dx2)
                dA2_dtau2 = dA2_dx2 * (-x2 / tau2)
                dexp2_dtau2 = exp2 * (x2 / tau2)
                dB2_dtau2 = dA2_dtau2 - dexp2_dtau2

                # derivatives of yields
                d_db0 = np.ones_like(t)
                d_db1 = A1
                d_db2 = B1
                d_db3 = B2

                dy_dtau1 = b1 * dA1_dtau1 + b2 * dB1_dtau1
                dy_dtau2 = b3 * dB2_dtau2

                d_dlogtau1 = tau1 * dy_dtau1
                d_dlogtau2 = tau2 * dy_dtau2

                H = np.column_stack([d_db0, d_db1, d_db2, d_db3, d_dlogtau1, d_dlogtau2])
                return H

            else:
                raise ValueError("model must be 'ns' or 'nss'")

    # -------------------------
    # Fit NS and NSS helpers
    # -------------------------
    @staticmethod
    def fit_ns_for_date(maturities, yields):
        maturities = np.array(maturities, dtype=float)
        yields = np.array(yields, dtype=float)

        def model(t, b0, b1, b2, tau):
            x = t / tau
            with np.errstate(divide='ignore', invalid='ignore'):
                L = (1 - np.exp(-x)) / x
                L = np.where(x == 0, 1.0, L)
            return b0 + b1*L + b2*(L - np.exp(-x))

        def loss(p):
            return np.sum((model(maturities, *p) - yields)**2)

        init = [yields.mean(), -1.0, 1.0, 2.5]
        bounds = [(None,None),(None,None),(None,None),(0.05,15.0)]

        res = minimize(loss, init, bounds=bounds)
        return tuple(res.x) if res.success else tuple(init)

    @staticmethod
    def fit_nss_given_taus(maturities, yields, tau1, tau2):
        t = np.array(maturities, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            x1 = t / tau1
            exp1 = np.exp(-x1)
            A1 = (1 - exp1) / x1
            A1 = np.where(x1 == 0, 1.0, A1)
            B1 = A1 - exp1

            x2 = t / tau2
            exp2 = np.exp(-x2)
            A2 = (1 - exp2) / x2
            A2 = np.where(x2 == 0, 1.0, A2)
            B2 = A2 - exp2

        X = np.column_stack([np.ones_like(t), A1, B1, B2])
        beta, _, _, _ = np.linalg.lstsq(X, yields, rcond=None)
        return tuple(beta)  # (b0,b1,b2,b3)

    # -------------------------
    # EKF + RTS smoother (upgraded, supports NS and NSS)
    # -------------------------
    def ekf_rts_smoother(self, yield_df, model='ns', Q_scale=1e-6, R_scale=1e-6,
                         phi=None, init_tau=None, init_tau2=None):
        """
        EKF+RTS for 'ns' (4-state) or 'nss' (6-state) models.
        - model: 'ns' or 'nss'
        - init_tau: for 'ns' a float; for 'nss' this is tau1 guess (float)
        - init_tau2: for 'nss' tau2 guess (float)
        - Q_scale, R_scale: either scalar or list-like length=k / N respectively
        Returns: (smoothed_states (T x k), diagnostics)
        """
        Y = yield_df.values
        maturities = yield_df.columns.values.astype(float)
        T, N = Y.shape

        if model == 'ns':
            k = 4
            if init_tau is None: init_tau = 2.5
        elif model == 'nss':
            k = 6
            if init_tau is None: init_tau = 2.5
            if init_tau2 is None: init_tau2 = 10.0
        else:
            raise ValueError("model must be 'ns' or 'nss'")

        F = np.eye(k) if phi is None else np.asarray(phi, dtype=float)

        # Q_scale and R_scale flex: accept scalar or per-state/per-observation list
        if np.isscalar(Q_scale):
            Q = np.eye(k) * float(Q_scale)
        else:
            Q = np.diag(np.asarray(Q_scale, dtype=float))
        if np.isscalar(R_scale):
            R = np.eye(N) * float(R_scale)
        else:
            R = np.diag(np.asarray(R_scale, dtype=float))

        # initial state x0
        y0 = Y[0].astype(float)
        if model == 'ns':
            beta_init = FixedIncomeToolkit.fit_ns_for_date(maturities, y0)
            x0 = np.zeros(k)
            x0[0:3] = beta_init[0:3]
            x0[3] = np.log(init_tau)
        else:  # nss
            b0,b1,b2,b3 = FixedIncomeToolkit.fit_nss_given_taus(maturities, y0, init_tau, init_tau2)
            x0 = np.zeros(k)
            x0[0:4] = np.array([b0,b1,b2,b3])
            x0[4] = np.log(init_tau)
            x0[5] = np.log(init_tau2)

        P0 = np.eye(k) * 1.0

        x_pred = np.zeros((T, k)); P_pred = np.zeros((T, k, k))
        x_filt = np.zeros((T, k)); P_filt = np.zeros((T, k, k))

        x_prev = x0.copy(); P_prev = P0.copy()
        for t_idx in range(T):
            # Predict
            x_prior = F @ x_prev
            P_prior = F @ P_prev @ F.T + Q

            # Nonlinear measurement and Jacobian
            y_prior = FixedIncomeToolkit.ns_yields_from_state(x_prior, maturities, model=model)
            H = FixedIncomeToolkit.ns_jacobian_wrt_state(x_prior, maturities, model=model)

            S = H @ P_prior @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            K = P_prior @ H.T @ S_inv

            y_obs = Y[t_idx]
            innov = y_obs - y_prior
            x_upd = x_prior + K @ innov
            P_upd = (np.eye(k) - K @ H) @ P_prior

            x_pred[t_idx] = x_prior; P_pred[t_idx] = P_prior
            x_filt[t_idx] = x_upd; P_filt[t_idx] = P_upd

            x_prev = x_upd; P_prev = P_upd

        # RTS smoother (backward pass)
        x_smooth = np.zeros_like(x_filt); P_smooth = np.zeros_like(P_filt)
        x_smooth[-1] = x_filt[-1]; P_smooth[-1] = P_filt[-1]
        for t_idx in range(T-2, -1, -1):
            P_f = P_filt[t_idx]; P_p1 = P_pred[t_idx+1]
            try:
                inv_Pp1 = np.linalg.inv(P_p1)
            except np.linalg.LinAlgError:
                inv_Pp1 = np.linalg.pinv(P_p1)
            C = P_f @ F.T @ inv_Pp1
            x_smooth[t_idx] = x_filt[t_idx] + C @ (x_smooth[t_idx+1] - x_pred[t_idx+1])
            P_smooth[t_idx] = P_f + C @ (P_smooth[t_idx+1] - P_p1) @ C.T

        diagnostics = {"x_pred": x_pred, "P_pred": P_pred, "x_filt": x_filt, "P_filt": P_filt}
        return x_smooth, diagnostics

    # -------------------------
    # VAR forecasting & reconstruction (supports NS/NSS)
    # -------------------------
    def var_forecast_reconstruct(self, smoothed_states, maturities, n_steps=10, n_scenarios=100, seed=1, model='ns'):
        """
        Fit VAR(1) on smoothed_states (T x k) and simulate scenarios forward.
        - model: 'ns' or 'nss' (affects reconstruction)
        Returns: scenarios array shape (n_scenarios, n_steps, n_maturities) and forecast_mean (n_steps x n_maturities)
        """
        rng = np.random.default_rng(seed)
        T, k = smoothed_states.shape
        maturities = np.array(maturities, dtype=float)

        var_model = VAR(smoothed_states)
        var_res = var_model.fit(maxlags=1)
        # coefs: (nlags, k, k) - for nlags=1 it's (1,k,k)
        coefs = var_res.coefs[0]  # shape (k,k)
        # intercept: length k
        try:
            intercept = var_res.intercept
        except Exception:
            # fallback: compute intercept from params
            intercept = var_res.params[0, :].astype(float) if hasattr(var_res, 'params') else np.zeros(k)
        resid_cov = np.cov(var_res.resid.T)

        scenarios = np.zeros((n_scenarios, n_steps, len(maturities)))
        for s in range(n_scenarios):
            x = smoothed_states[-1].copy()
            for t in range(n_steps):
                x = intercept + coefs @ x + rng.multivariate_normal(np.zeros(k), resid_cov)
                # reconstruct yields according to model
                scenarios[s, t] = FixedIncomeToolkit.ns_yields_from_state(x, maturities, model=model)
        forecast_mean = scenarios.mean(axis=0)
        return scenarios, forecast_mean

    # -------------------------
    # PCA, bootstrap, bonds, scenarios, CVaR, backtest (kept and cleaned)
    # -------------------------
    def pca_yield_changes(self, yield_df, n_components=3):
        dy = yield_df.diff().dropna()
        X = dy.values - np.mean(dy.values, axis=0)
        U,S,Vt = svd(X, full_matrices=False)
        components = Vt[:n_components,:]
        explained = (S**2) / np.sum(S**2)
        scores = X @ components.T
        return {"components":components, "explained":explained[:n_components], "scores":scores, "maturities":yield_df.columns.tolist()}

    def bootstrap_zero_curve_from_points(self, tenor_points):
        if isinstance(tenor_points, dict):
            items = sorted(tenor_points.items())
        else:
            items = sorted(tenor_points)
        T = np.array([i[0] for i in items]); y = np.array([i[1] for i in items])
        def zero_rate_func(tq):
            tq = np.array(tq, dtype=float)
            return np.interp(tq, T, y, left=y[0], right=y[-1])
        return zero_rate_func

    def add_bond(self, name, face, coupon_rate, maturity, freq=1, ns_params=None):
        if ns_params is None: ns_params = (0.02,0.0,0.0,2.5)
        self.bonds.append({"name":name,"face":face,"coupon_rate":coupon_rate,"maturity":maturity,"freq":freq,"ns_params":ns_params})

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

    def simulate_ns_factor_shocks(self, n_scenarios=2000, level_vol=0.008, slope_vol=0.004, curvature_vol=0.003, seed=42):
        rng = np.random.default_rng(seed)
        shocks = rng.normal(loc=0.0, scale=[level_vol, slope_vol, curvature_vol], size=(n_scenarios,3))
        self.ns_factor_shocks = shocks
        return shocks

    def generate_scenario_returns_multi_factor(self, horizon_years=1/12, n_scenarios=2000,
                                             level_vol=0.008, slope_vol=0.004, curvature_vol=0.003,
                                             seed=42, include_spread=False, spread_vol=0.001):
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
                if include_spread:
                    spread_shift = np.random.default_rng(seed+i+j).normal(0, spread_vol)
                    price = price * np.exp(-spread_shift * b["maturity"])
                total_return = (price + coupon_received - b["face"]) / b["face"]
                scenario_returns[i,j] = total_return
        self.scenario_returns = scenario_returns
        return scenario_returns

    def minimize_cvar(self, alpha=0.95, target_return=None, bounds=None, target_duration=None, duration_tol=None):
        R = self.scenario_returns
        if R is None:
            raise ValueError("No scenario returns generated.")
        T,N = R.shape
        num_vars = N + 1 + T
        c = np.zeros(num_vars); c[N] = 1.0; c[N+1:] = 1.0 / ((1-alpha)*T)
        A_ub = np.zeros((2*T, num_vars)); b_ub = np.zeros(2*T)
        for i in range(T):
            A_ub[i, :N] = -R[i]; A_ub[i, N] = -1.0; A_ub[i, N+1+i] = -1.0
        for i in range(T):
            A_ub[T+i, N+1+i] = -1.0
        A_eq = np.zeros((1, num_vars)); A_eq[0,:N] = 1.0; b_eq = np.array([1.0])
        if target_return is not None:
            ER = np.mean(R, axis=0)
            row = np.zeros(num_vars); row[:N] = -ER
            A_ub = np.vstack([A_ub, row]); b_ub = np.append(b_ub, -target_return)
        if target_duration is not None:
            df_prices = self.price_all_bonds()
            durations = df_prices["modified_duration"].values
            if duration_tol is None:
                A_eq = np.vstack([A_eq, np.zeros((1, num_vars))])
                A_eq[-1, :N] = durations
                b_eq = np.append(b_eq, target_duration)
            else:
                row_up = np.zeros(num_vars); row_up[:N] = durations
                A_ub = np.vstack([A_ub, row_up]); b_ub = np.append(b_ub, target_duration + duration_tol)
                row_lo = np.zeros(num_vars); row_lo[:N] = -durations
                A_ub = np.vstack([A_ub, row_lo]); b_ub = np.append(b_ub, -(target_duration - duration_tol))
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
    # Backtest wrapper (kept)
    # -------------------------
    def backtest_rolling(self, yield_series_df, maturities_years, window=252, rebalance_days=21,
                         alpha=0.95, target_duration=None, duration_tol=None, tc=0.0005):
        dates = yield_series_df.index
        steps = max(0, (len(dates) - window) // rebalance_days)
        weights_hist=[]; perf_hist=[]; prev_w = None
        for k in range(steps):
            start = k*rebalance_days
            train = yield_series_df.iloc[start:start+window]
            test = yield_series_df.iloc[start+window:start+window+rebalance_days]
            if len(test)==0: break
            params = self.fit_ns_for_date(maturities_years, train.iloc[-1].values)
            for b in self.bonds: b["ns_params"] = params
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
            opt,res,extras = self.minimize_cvar(alpha=alpha, target_return=None, bounds=None, target_duration=target_duration, duration_tol=duration_tol)
            if opt is None:
                w = np.ones(len(self.bonds))/len(self.bonds)
            else:
                w = opt["weights"]
            turnover = np.sum(np.abs(w - (prev_w if prev_w is not None else np.zeros_like(w))))
            tc_cost = tc * turnover
            realized = np.mean(scenarios, axis=0) @ w - tc_cost
            weights_hist.append(w); perf_hist.append(realized)
            prev_w = w.copy()
        return {"strategy": np.array(perf_hist), "weights":np.array(weights_hist)}

    # -------------------------
    # Convenience: fit_ns_time_series used by demo/backtest
    # -------------------------
    def fit_ns_time_series(self, yield_df, maturities_years):
        records=[]
        for date, row in yield_df.iterrows():
            params = FixedIncomeToolkit.fit_ns_for_date(maturities_years, row.values.astype(float))
            records.append((date,)+params)
        return pd.DataFrame(records, columns=["Date","beta0","beta1","beta2","tau"]).set_index("Date")
