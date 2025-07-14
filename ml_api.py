from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_from_model import predict_from_saved_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "ML Server is awake"}

@app.get("/predict/{product_id}")
def predict(product_id: str):
    try:
        result = predict_from_saved_model(product_id)
        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"]}
        return result
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {"error": str(e)}