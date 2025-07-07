# predict_from_model.py

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import joblib

def predict_from_saved_model(product_id):
    # 🔥 Use absolute path to avoid file-not-found errors
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", f"{product_id}.pkl")

    if not os.path.exists(model_path):
        return { "error": f"Model for {product_id} not found at {model_path}" }

    model = joblib.load(model_path)
    start_date = datetime.now().date() + timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=7)
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    next_week = forecast.tail(7)[['ds', 'yhat']]
    next_week['yhat'] = next_week['yhat'].clip(lower=0).round().astype(int)


    results = [
        { "date": row["ds"].strftime("%Y-%m-%d"), "predictedSales": int(row["yhat"]) }
        for _, row in next_week.iterrows()
    ]
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({ "error": "Product ID not provided" }))
        sys.exit(1)

    product_id = sys.argv[1]
    result = predict_from_saved_model(product_id)
    print(json.dumps(result))
