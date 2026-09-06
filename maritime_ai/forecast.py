import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - dependency is pinned in requirements.txt
    XGBRegressor = None


FEATURES = [
    "lag_1", "lag_2", "lag_4", "mean_4", "volatility_4",
    "bunker_usd_per_tonne", "bunker_change_pct", "coal_price_usd_per_tonne",
    "coal_change_pct", "usd_inr", "fx_change_pct", "congestion_days",
    "route_distance_nm", "vessel_class_code", "month", "monsoon_indicator",
]

MODEL_VERSIONS = {
    "Ridge": "Ridge-v1.0",
    "XGBoost": "XGBoost-v1.0",
}


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().sort_values("date")
    rate = frame["rate_usd_per_tonne"]
    frame["lag_1"] = rate.shift(1)
    frame["lag_2"] = rate.shift(2)
    frame["lag_4"] = rate.shift(4)
    frame["mean_4"] = rate.shift(1).rolling(4).mean()
    frame["volatility_4"] = rate.shift(1).rolling(4).std()
    frame["bunker_change_pct"] = frame["bunker_usd_per_tonne"].pct_change() * 100
    frame["coal_change_pct"] = frame["coal_price_usd_per_tonne"].pct_change() * 100
    frame["fx_change_pct"] = frame["usd_inr"].pct_change() * 100
    frame["month"] = pd.to_datetime(frame["date"]).dt.month
    if "monsoon_indicator" not in frame:
        frame["monsoon_indicator"] = frame["month"].between(6, 9).astype(int)
    return frame.dropna().reset_index(drop=True)


def make_model(model_name: str):
    """Return a deterministic model specification used by both backtest and forecast."""
    if model_name == "Ridge":
        return Ridge(alpha=1.0)

    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("XGBoost is required. Add xgboost to requirements.txt and install dependencies.")
        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=26006,
            n_jobs=2,
        )

    raise ValueError(f"Unknown model: {model_name}")


def _single_model_backtest(frame: pd.DataFrame, model_name: str, min_train: int) -> dict:
    predictions, actuals = [], []
    for point in range(min_train, len(frame)):
        train, test = frame.iloc[:point], frame.iloc[[point]]
        model = make_model(model_name).fit(train[FEATURES], train["rate_usd_per_tonne"])
        predictions.append(float(model.predict(test[FEATURES])[0]))
        actuals.append(float(test["rate_usd_per_tonne"].iloc[0]))

    errors = np.array(predictions) - np.array(actuals)
    mae = float(np.abs(errors).mean()) if len(errors) else float("nan")
    rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan")
    return {
        "model": model_name,
        "version": MODEL_VERSIONS[model_name],
        "mae": mae,
        "rmse": rmse,
        "n_predictions": len(predictions),
        "predictions": predictions,
        "actuals": actuals,
    }


def backtest(data: pd.DataFrame, min_train: int = 52) -> dict:
    """Expanding-window one-step backtest for Ridge and XGBoost.

    The model with the lower walk-forward MAE is selected for production forecasting.
    If MAE ties, RMSE is used as the tie-breaker.
    """
    frame = make_features(data)
    if len(frame) <= min_train:
        raise ValueError("Not enough observations for the requested walk-forward backtest.")

    ridge = _single_model_backtest(frame, "Ridge", min_train)
    xgb = _single_model_backtest(frame, "XGBoost", min_train)
    comparison = pd.DataFrame([
        {"Model": ridge["model"], "Version": ridge["version"], "MAE": ridge["mae"], "RMSE": ridge["rmse"]},
        {"Model": xgb["model"], "Version": xgb["version"], "MAE": xgb["mae"], "RMSE": xgb["rmse"]},
    ])

    winner = min((ridge, xgb), key=lambda r: (r["mae"], r["rmse"]))
    return {
        **winner,
        "selected_model": winner["model"],
        "selected_version": winner["version"],
        "comparison": comparison,
        "ridge": ridge,
        "xgboost": xgb,
    }


def forecast(
    data: pd.DataFrame,
    horizon_weeks: int = 12,
    driver_changes: dict | None = None,
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    """Recursive freight forecast using the selected validated model.

    ``model_name`` should normally be the winner from ``backtest()``. It is
    exposed explicitly so scenario/decision analysis uses the same model as
    the audited production path instead of silently changing models.
    """
    driver_changes = driver_changes or {}
    history = data.copy().sort_values("date").reset_index(drop=True)
    model_data = make_features(history)
    model = make_model(model_name).fit(model_data[FEATURES], model_data["rate_usd_per_tonne"])

    rows = []
    for _ in range(horizon_weeks):
        last = history.iloc[-1]
        date = last.date + pd.Timedelta(days=7)
        next_month = date.month
        next_row = {
            "date": date,
            "route": last.get("route", "Selected route"),
            "bunker_usd_per_tonne": history["bunker_usd_per_tonne"].tail(4).mean() * (1 + driver_changes.get("bunker_pct", 0) / 100),
            "coal_price_usd_per_tonne": history["coal_price_usd_per_tonne"].tail(4).mean() * (1 + driver_changes.get("coal_pct", 0) / 100),
            "usd_inr": history["usd_inr"].tail(4).mean() * (1 + driver_changes.get("fx_pct", 0) / 100),
            "congestion_days": max(0, history["congestion_days"].tail(4).mean() + driver_changes.get("congestion_days", 0)),
            "route_distance_nm": last.get("route_distance_nm", 2400.0),
            "vessel_class_code": last.get("vessel_class_code", 1.0),
            "monsoon_indicator": int(6 <= next_month <= 9),
            "rate_usd_per_tonne": np.nan,
        }
        temporary = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        featured = make_features(temporary)
        next_row["rate_usd_per_tonne"] = float(model.predict(featured.iloc[[-1]][FEATURES])[0])
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        rows.append(next_row)

    result = pd.DataFrame(rows)
    result["model"] = model_name
    result["model_version"] = MODEL_VERSIONS[model_name]
    return result
