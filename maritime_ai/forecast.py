import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


FEATURES = ["lag_1", "lag_2", "lag_4", "mean_4", "bunker_usd_per_tonne", "congestion_days"]


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().sort_values("date")
    rate = frame["rate_usd_per_tonne"]
    frame["lag_1"] = rate.shift(1)
    frame["lag_2"] = rate.shift(2)
    frame["lag_4"] = rate.shift(4)
    frame["mean_4"] = rate.shift(1).rolling(4).mean()
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


def forecast(data: pd.DataFrame, horizon_weeks: int = 12) -> pd.DataFrame:
    history = data.copy().sort_values("date").reset_index(drop=True)
    model_data = make_features(history)
    model = Ridge(alpha=1.0).fit(model_data[FEATURES], model_data["rate_usd_per_tonne"])
    rows = []
    for _ in range(horizon_weeks):
        last = history.iloc[-1]
        date = last.date + pd.Timedelta(days=7)
        # Hold operational drivers at recent four-week averages; rate lags recurse.
        next_row = {"date": date, "route": last.get("route", "Selected route"),
                    "bunker_usd_per_tonne": history["bunker_usd_per_tonne"].tail(4).mean(),
                    "congestion_days": history["congestion_days"].tail(4).mean(),
                    "rate_usd_per_tonne": np.nan}
        temporary = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        featured = make_features(temporary)
        next_row["rate_usd_per_tonne"] = float(model.predict(featured.iloc[[-1]][FEATURES])[0])
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        rows.append(next_row)
    return pd.DataFrame(rows)
