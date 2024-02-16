from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import threading
import time
import requests

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin, you can replace it with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow GET, POST, and OPTIONS methods
    allow_headers=["*"],
)

# Global variable to control recording state
is_recording = False

# Load the machine failure detection model
with open("model/machine_failure_detection_model3.pkl", 'rb') as f:
    model = pickle.load(f)

# Define your model class
class Input(BaseModel):
    RMS: float
    Mean: float
    MA1: float
    MA2: float
    MA3: float
    F1: float
    F2: float
    F3: float

# Endpoint to start recording
@app.post("/record")
def start_recording():
    global is_recording
    if not is_recording:
        is_recording = True
        threading.Thread(target=record).start()
        return {"message": "Recording started successfully"}
    else:
        return {"message": "Recording is already in progress"}

# Endpoint to stop recording
@app.post("/stop")
def stop_recording():
    global is_recording
    is_recording = False
    return {"message": "Recording stopped successfully"}

# Function to perform recording
def record():
    # Your recording logic here
    while is_recording:
        print("Recording...")
        # Simulated recording activity
        time.sleep(1)
    print("Recording stopped.")

# Endpoint to predict failure
@app.post("/predict")
def predict_failure(input: Input):
    try:
        # Get data from POST request
        data = input.dict()

        # Convert data to pandas DataFrame
        data_df = pd.DataFrame([data])

        # Get the model's prediction
        prediction = model.predict(data_df)

        # Return prediction as JSON
        feedback = {"prediction": prediction[0], "message": "Prediction successful"}

        # Make a POST request back to sender machine with feedback
        sender_url = "http://10.10.14.14:3001"  # Replace with actual sender machine IP and port
        response = requests.post(sender_url, json=feedback)
        if response.ok:
            print("Feedback sent successfully to sender machine")
        else:
            print("Failed to send feedback to sender machine")

        return feedback
    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
