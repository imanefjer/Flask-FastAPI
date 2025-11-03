from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

app = FastAPI()

class iris(BaseModel):
    a: float
    b: float
    c: float
    d: float

# Load the model using pickle
model = pickle.load(open('model_iris', 'rb'))

@app.get("/")
def home():
    return {'message': 'ML model for Iris prediction'}

@app.post('/make_predictions')
async def make_predictions(features: iris):
    prediction = model.predict([[features.a, features.b, features.c, features.d]])[0]
    return {"prediction": str(prediction)}

if __name__ == "__main__":  # Fixed: removed extra