# importing the necessary dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn 
import pickle
import numpy as np
import pandas as pd

# loading the saved model
model=pickle.load(open("C:\\Users\\Akind\\Documents\\ML App\\model\\machine_failure_detection_model3.pkl",'rb'))

app = FastAPI()

class Input(BaseModel):
    RMS: float
    Mean: float
    MA1: float
    MA2: float
    MA3: float
    F1: float
    F2: float
    F3: float


@app.get("/")
def read_root():
    return {"msg": "Machine Failure Predictor"}

@app.post("/predict")
def predict_failure(input:Input):
    data = input.dict() # Get data from POST request
    print(data)
    data_df = pd.DataFrame([data])  # Convert to pandas DataFrame
    print(data_df)
    prediction = model.predict([data_df.iloc[0]]) # Get the model's prediction
    print(prediction)
    return {"prediction": prediction[0]} # Return prediction as JSON

@app.post("/receive-request")
def receive_request():
    print("Received HTTP request from sending machine")
    # Add any processing logic here
    return {"status": "success", "message": "Request received successfully"}

if __name__ == "__main__":
    uvicorn.run(app,port=8000)


