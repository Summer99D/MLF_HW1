import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit



def compute_r2s(df, target_col="CRSP_SPvw_minus_Rfree", start_oos="1965-01-01"):
    df = df.copy()

    # Ensure datetime index
    if "date" not in df.columns:
        if "yyyymm" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
            except:
                df["date"] = pd.to_datetime(df["yyyymm"])
        else:
            raise ValueError("No 'date' or 'yyyymm' column found.")
    df.set_index("date", inplace=True)

    predictor_cols = [col for col in df.columns if col.endswith("_lag1")]

    # Check for missing values
    if df[predictor_cols + [target_col]].isnull().any().any():
        raise ValueError("Missing values detected in predictors or target.")

    in_sample_r2 = {}
    oos_r2 = {}

    # Get index to start OOS
    start_idx = df.index.get_indexer([pd.to_datetime(start_oos)], method="bfill")[0]

    for predictor in predictor_cols:
        X = df[[predictor]].values
        y = df[target_col].values

        # In-sample with sklearn
        model_in = LinearRegression().fit(X, y)
        y_pred_in = model_in.predict(X)
        in_sample_r2[predictor] = r2_score(y, y_pred_in)

        # Out-of-sample expanding window
        y_true, y_pred = [], []

        for i in range(start_idx, len(df)):
            if i < 60:
                continue

            X_train = df.iloc[:i][[predictor]].values
            y_train = df.iloc[:i][target_col].values
            X_test = df[[predictor]].iloc[i : i + 1].values
            y_test = df[target_col].iloc[i]

            model = LinearRegression().fit(X_train, y_train)
            pred = model.predict(X_test)[0]

            y_true.append(y_test)
            y_pred.append(pred)

        if len(y_true) > 1:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            sse = np.sum((y_true - y_pred) ** 2)
            sst = np.sum((y_true - np.mean(y_true)) ** 2)
            oos_r2[predictor] = 1 - sse / sst if sst != 0 else np.nan
        else:
            oos_r2[predictor] = np.nan

    return pd.DataFrame(
        {"In-Sample R^2": in_sample_r2, "Out-of-Sample R^2": oos_r2}
    ).sort_values(by="In-Sample R^2", ascending=False)


def regression_with_regulariser(
    df,
    target_col="CRSP_SPvw_minus_Rfree",
    start_oos="1965-01-01",
    regulariser="OLS",
    plot=False,
    min_train_size=60,
):
    df = df.copy()

    if "date" not in df.columns:
        if "yyyymm" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
            except:
                df["date"] = pd.to_datetime(df["yyyymm"])
        else:
            raise ValueError("Missing date/yyyymm column")
    df.set_index("date", inplace=True)

    predictor_cols = [col for col in df.columns if col.endswith("_lag1")]
    if len(predictor_cols) == 0:
        raise ValueError("No predictors ending with '_lag1' found.")

    if df[predictor_cols + [target_col]].isnull().any().any():
        raise ValueError("Missing values in predictors or target.")

    X = df[predictor_cols]
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), index=X.index, columns=predictor_cols
    )

    def fit_model(X_train, y_train, method):
        method = method.upper()
        alphas = np.logspace(-3, 0, 10)
        if method == "OLS":
            return LinearRegression().fit(X_train, y_train)
        elif method == "LASSO":
            return LassoCV(cv=5, alphas=alphas, n_jobs=-1).fit(X_train, y_train)
        elif method == "RIDGE":
            return RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
        elif method == "ELASTICNET":
            return ElasticNetCV(cv=5, alphas=alphas, n_jobs=-1).fit(X_train, y_train)
        else:
            raise ValueError("Unknown regulariser.")

    # In-sample fit
    model_full = fit_model(X_scaled, y, regulariser)
    in_sample_preds = model_full.predict(X_scaled)
    in_sample_r2 = r2_score(y, in_sample_preds)

    # Expanding OOS
    start_idx = X_scaled.index.get_indexer([pd.to_datetime(start_oos)], method="bfill")[
        0
    ]
    oos_dates, oos_y, oos_preds, cum_r2_list = [], [], [], []

    for i in range(start_idx, len(X_scaled)):
        if i < min_train_size:
            continue

        X_train = X_scaled.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X_scaled.iloc[i : i + 1]
        y_test = y.iloc[i]

        model = fit_model(X_train, y_train, regulariser)
        pred = model.predict(X_test)[0]

        oos_dates.append(X_scaled.index[i])
        oos_y.append(y_test)
        oos_preds.append(pred)

        if len(oos_y) > 1:
            y_true_arr = np.array(oos_y)
            y_pred_arr = np.array(oos_preds)
            sse = np.sum((y_true_arr - y_pred_arr) ** 2)
            sst = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
            cum_r2 = 1 - sse / sst if sst != 0 else np.nan
            cum_r2_list.append(cum_r2)
        else:
            cum_r2_list.append(np.nan)

    summary_df = pd.DataFrame(
        {
            "Method": [regulariser.upper()],
            "In-Sample R^2": [in_sample_r2],
            "Out-of-Sample R^2": [cum_r2_list[-1] if cum_r2_list else np.nan],
        }
    )

    oos_ts_df = pd.DataFrame({"OOS_Cumulative_R^2": cum_r2_list}, index=oos_dates)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(oos_ts_df.index, oos_ts_df["OOS_Cumulative_R^2"], marker="o")
        plt.xlabel("Date")
        plt.ylabel("Cumulative OOS R^2")
        plt.title(f"Cumulative OOS R^2 Over Time ({regulariser.upper()})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return summary_df



# ---------------------------------------------------------------------------
def _ridge_closed_form(XTX, XTy, alpha):
    """w = (XᵀX + αI)⁻¹ Xᵀy  (closed‑form ridge coefficients)."""
    p = XTX.shape[0]
    return np.linalg.solve(XTX + alpha * np.eye(p), XTy)


def rbf_feature_sweep_fast(
    df,
    feature_counts=(10, 25, 50, 100, 200, 400),
    alpha_grid=np.logspace(-4, 4, 25),
    start_year=1965,
    init_years=10,
    gamma=1.0,
    random_state=0,
):
    """
    Much faster expanding‑window OOS R² for several RBF feature counts.
    - init_years: how many years to use for the initial α selection.
    """
    # ---------- tidy index ----------
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"]))
        elif "yyyymm" in df.columns:
            df = df.set_index(pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m"))
        else:
            raise ValueError("Need a datetime index, 'date', or 'yyyymm' column.")
    y = df["CRSP_SPvw_minus_Rfree"]
    X_raw = df.drop(
        columns=["CRSP_SPvw_minus_Rfree", "yyyymm", "date"], errors="ignore"
    )

    r2_dict = {}
    for n in feature_counts:
        # --- 1) one‑time RBF transform -------------------------------------
        rbf = RBFSampler(n_components=n, gamma=gamma, random_state=random_state)
        X_rbf = pd.DataFrame(rbf.fit_transform(X_raw), index=X_raw.index)
        X = X_rbf.values
        y_arr = y.values

        # --- 2) choose α once on the first `init_years` ---------------------
        split_idx = np.searchsorted(
            y.index, y.index[0] + pd.DateOffset(years=init_years)
        )
        ridge_cv = RidgeCV(alphas=alpha_grid, cv=5).fit(
            X[:split_idx], y_arr[:split_idx]
        )
        alpha = ridge_cv.alpha_

        # --- 3) pre‑allocate cumulative matrices ---------------------------
        p = X.shape[1]
        XTX = np.zeros((p, p))
        XTy = np.zeros(p)

        preds, truth, bench = [], [], []
        start_idx = np.searchsorted(y.index, pd.Timestamp(f"{start_year}-01-01"))

        for t in range(len(y)):  # incremental update
            xt = X[t][:, None]  # column‑vector
            yt = y_arr[t]

            # update sufficient statistics
            XTX += xt @ xt.T
            XTy += xt.flatten() * yt

            if t >= start_idx:
                w = _ridge_closed_form(XTX, XTy, alpha)
                preds.append((xt.T @ w)[0])
                truth.append(yt)
                bench.append(y_arr[:t].mean())

        r2 = 1 - np.sum((np.array(truth) - np.array(preds)) ** 2) / np.sum(
            (np.array(truth) - np.array(bench)) ** 2
        )
        r2_dict[n] = r2

    # ---------- plot -------------------------------------------------------
    plt.figure()
    plt.plot(r2_dict.keys(), r2_dict.values(), marker="o")
    plt.xlabel("# RBF features")
    plt.ylabel("OOS $R^2$")
    plt.title("Part (c) – fast expanding‑window performance")
    plt.grid(True)
    plt.show()
    return r2_dict





# ────────────────────────────────────────────────────────────────────────────────
# small helpers (shared by both fast functions)
# ────────────────────────────────────────────────────────────────────────────────


def _prep_df(df):
    """Ensure DatetimeIndex and return y, X_raw."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if "yyyymm" in df.columns:
            df = (
                df.assign(date=pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m"))
                .set_index("date")
                .sort_index()
            )
        else:
            raise ValueError("DataFrame needs a DatetimeIndex or a 'yyyymm' column.")
    y = df["CRSP_SPvw_minus_Rfree"]
    X_raw = df.drop(columns=["CRSP_SPvw_minus_Rfree", "yyyymm"], errors="ignore")
    return y, X_raw


# ────────────────────────────────────────────────────────────────────────────────
# (d) FAST rolling‑window effect
# ────────────────────────────────────────────────────────────────────────────────
def rolling_window_effect_fast(
    df,
    best_n,
    windows=(12, 36, 60, 120),
    alpha_grid=np.logspace(-4, 4, 25),
    gamma=1.0,
    random_state=0,
):
    """
    MUCH faster OOS R² vs. rolling‑window length.
    Chooses α once on the first window, then uses rank‑1 updates.
    """
    y, X_raw = _prep_df(df)

    # one‑time RBF transform
    rbf = RBFSampler(n_components=best_n, gamma=gamma, random_state=random_state)
    X_rbf = pd.DataFrame(rbf.fit_transform(X_raw), index=X_raw.index)
    X = X_rbf.values
    y_arr = y.values
    p = X.shape[1]

    r2_roll = {}
    for w in windows:
        # ---------- choose α once on first window ----------
        ridge_cv = RidgeCV(alphas=alpha_grid, cv=5).fit(X[:w], y_arr[:w])
        alpha = ridge_cv.alpha_

        # ---------- initialise sufficient statistics ----------
        XTX = X[:w].T @ X[:w]
        XTy = X[:w].T @ y_arr[:w]

        preds, truth, bench = [], [], []
        for t in range(w, len(y)):
            # remove oldest obs, add newest obs (rank‑1 updates)
            x_out = X[t - w]  # leaving the window
            y_out = y_arr[t - w]
            XTX -= np.outer(x_out, x_out)
            XTy -= x_out * y_out

            x_in = X[t]  # entering the window
            y_in = y_arr[t]
            XTX += np.outer(x_in, x_in)
            XTy += x_in * y_in

            w_coef = _ridge_closed_form(XTX, XTy, alpha)
            preds.append(x_in @ w_coef)
            truth.append(y_in)
            bench.append(y_arr[t - w + 1 : t + 1].mean())  # rolling mean

        r2_roll[w] = 1 - np.sum((np.array(truth) - np.array(preds)) ** 2) / np.sum(
            (np.array(truth) - np.array(bench)) ** 2
        )

    # ---------- plot ----------
    plt.figure()
    plt.bar([str(w) for w in r2_roll], r2_roll.values())
    plt.xlabel("window (months)")
    plt.ylabel("OOS $R^2$")
    plt.title(f"Part (d): rolling‑window (n={best_n}) – FAST")
    plt.show()
    return r2_roll


# ────────────────────────────────────────────────────────────────────────────────
# (e) FAST CV‑fold effect
# ────────────────────────────────────────────────────────────────────────────────
def cv_fold_effect_fast(
    df,
    best_n,
    folds=(3, 5, 10),
    alpha_grid=np.logspace(-4, 4, 25),
    start_year=1965,
    gamma=1.0,
    random_state=0,
):
    """
    MUCH faster OOS R² vs. # CV folds (expanding window).
    Picks α once (via each fold value) on initial training chunk, then
    updates coefficients analytically as the sample expands.
    """
    y, X_raw = _prep_df(df)

    # one‑time RBF transform
    rbf = RBFSampler(n_components=best_n, gamma=gamma, random_state=random_state)
    X_rbf = pd.DataFrame(rbf.fit_transform(X_raw), index=X_raw.index)
    X = X_rbf.values
    y_arr = y.values
    p = X.shape[1]

    start_idx = np.searchsorted(y.index, pd.Timestamp(f"{start_year}-01-01"))

    r2_fold = {}
    for k in folds:
        # ---------- choose α once on initial training sample ----------
        ridge_cv = RidgeCV(alphas=alpha_grid, cv=k).fit(
            X[:start_idx], y_arr[:start_idx]
        )
        alpha = ridge_cv.alpha_

        # ---------- initialise cumulative matrices ----------
        XTX = X[:start_idx].T @ X[:start_idx]
        XTy = X[:start_idx].T @ y_arr[:start_idx]

        preds, truth, bench = [], [], []
        for t in range(start_idx, len(y)):
            x_new = X[t][:, None]  # column‑vector
            y_new = y_arr[t]

            # expand cumulative matrices
            XTX += x_new @ x_new.T
            XTy += x_new.flatten() * y_new

            w_coef = _ridge_closed_form(XTX, XTy, alpha)
            preds.append((x_new.T @ w_coef)[0])
            truth.append(y_new)
            bench.append(y_arr[:t].mean())  # historical mean benchmark

        r2_fold[k] = 1 - np.sum((np.array(truth) - np.array(preds)) ** 2) / np.sum(
            (np.array(truth) - np.array(bench)) ** 2
        )

    # ---------- plot ----------
    plt.figure()
    plt.plot(r2_fold.keys(), r2_fold.values(), marker="s")
    plt.xlabel("CV folds")
    plt.ylabel("OOS $R^2$")
    plt.grid(True)
    plt.title(f"Part (e): CV‑fold effect (n={best_n}) – FAST")
    plt.show()
    return r2_fold


# ────────────────────────────────────────────────────────────────────────────────
# (g) compare alternative regressors (OOS R²)
# ────────────────────────────────────────────────────────────────────────────────

def fast_compare_regressors_oos_r2(df, alpha=1.0, gamma=0.1, n_components=5):
    y = df["CRSP_SPvw_minus_Rfree"].values
    X = df.drop(columns=["CRSP_SPvw_minus_Rfree", "yyyymm"], errors="ignore").values
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "Ridge (Baseline)": make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        "Kernel Ridge (RBF)": make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)),
        "PCA + Ridge": make_pipeline(StandardScaler(), PCA(n_components=n_components), Ridge(alpha=alpha)),
        "PLS Regression": make_pipeline(StandardScaler(), PLSRegression(n_components=n_components)),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3)
    }

    results = {}

    for name, model in models.items():
        r2_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            fitted_model = clone(model).fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))

        results[name] = np.mean(r2_scores)

    return pd.DataFrame.from_dict(results, orient='index', columns=["OOS R^2"]).sort_values(by="OOS R^2", ascending=False)

