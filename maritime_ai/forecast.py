import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


FEATURES = [
    "lag_1", "lag_2", "lag_4", "mean_4", "volatility_4",
    "bunker_usd_per_tonne", "bunker_change_pct", "coal_price_usd_per_tonne",
    "coal_change_pct", "usd_inr", "fx_change_pct", "congestion_days",
    "route_distance_nm", "vessel_class_code", "month", "monsoon_indicator",
]


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


def backtest(data: pd.DataFrame, min_train: int = 52) -> dict:
    """Expanding-window one-step backtest; no future rate leaks into features."""
    frame = make_features(data)
    predictions, actuals = [], []
    for point in range(min_train, len(frame)):
        train, test = frame.iloc[:point], frame.iloc[[point]]
        model = Ridge(alpha=1.0).fit(train[FEATURES], train["rate_usd_per_tonne"])
        predictions.append(float(model.predict(test[FEATURES])[0]))
        actuals.append(float(test["rate_usd_per_tonne"].iloc[0]))
    errors = np.array(predictions) - np.array(actuals)
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "n_predictions": len(predictions), "predictions": predictions, "actuals": actuals}


def forecast(data: pd.DataFrame, horizon_weeks: int = 12, driver_changes: dict | None = None) -> pd.DataFrame:
    """Recursive Ridge forecast with scenario-adjustable market drivers.

    ``driver_changes`` contains percentage changes for bunker, coal and FX, plus
    additional congestion days. It changes forecast inputs before prediction.
    """
    driver_changes = driver_changes or {}
    history = data.copy().sort_values("date").reset_index(drop=True)
    model_data = make_features(history)
    model = Ridge(alpha=1.0).fit(model_data[FEATURES], model_data["rate_usd_per_tonne"])
    rows = []
    for _ in range(horizon_weeks):
        last = history.iloc[-1]
        date = last.date + pd.Timedelta(days=7)
        # Hold operational drivers at recent four-week averages; rate lags recurse.
        next_month = date.month
        next_row = {"date": date, "route": last.get("route", "Selected route"),
                    "bunker_usd_per_tonne": history["bunker_usd_per_tonne"].tail(4).mean() * (1 + driver_changes.get("bunker_pct", 0) / 100),
                    "coal_price_usd_per_tonne": history["coal_price_usd_per_tonne"].tail(4).mean() * (1 + driver_changes.get("coal_pct", 0) / 100),
                    "usd_inr": history["usd_inr"].tail(4).mean() * (1 + driver_changes.get("fx_pct", 0) / 100),
                    "congestion_days": max(0, history["congestion_days"].tail(4).mean() + driver_changes.get("congestion_days", 0)),
                    "route_distance_nm": last.get("route_distance_nm", 2400.0),
                    "vessel_class_code": last.get("vessel_class_code", 1.0),
                    "monsoon_indicator": int(6 <= next_month <= 9),
                    "rate_usd_per_tonne": np.nan}
        temporary = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        featured = make_features(temporary)
        next_row["rate_usd_per_tonne"] = float(model.predict(featured.iloc[[-1]][FEATURES])[0])
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        rows.append(next_row)
    return pd.DataFrame(rows)
