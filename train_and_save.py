# train_and_save.py

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
from prophet import Prophet
import joblib

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["walmart"]
sales_collection = db["sales"]
product_collection = db["products"]

def load_sales_data(product_id):
    data = list(sales_collection.find({"productId": ObjectId(product_id)}))
    if not data:
        print(f"No sales data for product {product_id}")
        return None

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    grouped = df.groupby('date').agg({'quantity': 'sum'}).reset_index()
    grouped.rename(columns={'date': 'ds', 'quantity': 'y'}, inplace=True)
    return grouped

def train_and_save_model(product_id):
    df = load_sales_data(product_id)
    if df is None or len(df) < 2:
        return

    model = Prophet()
    model.fit(df)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{product_id}.pkl")
    print(f"✅ Model saved for product {product_id}")

if __name__ == "__main__":
    products = list(product_collection.find({}))
    for product in products:
        train_and_save_model(str(product["_id"]))
