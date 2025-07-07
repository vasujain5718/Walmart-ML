from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_from_model import predict_from_saved_model

app = FastAPI()

# Allow all origins (for dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/{product_id}")
def predict(product_id: str):
    result = predict_from_saved_model(product_id)
    return result