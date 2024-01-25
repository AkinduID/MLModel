# import required libraries
import streamlit as st 
import requests
import json
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftpk
from matplotlib import pyplot as plt

import threading
import time

import sounddevice as sd
import soundfile as sf
import librosa

from IPython.display import Image
import ipywidgets as widgets
from IPython.display import display

# Flag to control recording state
is_rec = True

def send_error_request(error_type):
    # Replace the following URL with the actual endpoint of your web server
    server_url = "https://your-web-server.com/error-handler"
    
    # Send HTTP POST request with error type in the payload
    payload = {"error_type": error_type}
    response = requests.post(server_url, data=payload)
    
    # Check the response status
    if response.status_code == 200:
        st.success(f"Error reported successfully: {error_type}")
    else:
        st.error(f"Failed to report error: {error_type}")
        st.write(f"Response status code: {response.status_code}")

# This fuction records the audio and gives predictions for the audio
def predict():
    # Recording the audio
    samplerate = 48000  
    duration = 5 # seconds
    audio = 'test.wav'
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    sd.wait()
    sf.write(audio, mydata, samplerate)

    # Reading the audio
    s_rate, signal = wavfile.read(audio)

    # Extracting the features
    mean = abs(signal).mean()
    rms = np.sqrt(abs(signal**2).mean())
    FFT_ = abs(fftpk.fft(signal))
    FFT = FFT_[range(len(FFT_)//2)]
    freqs = fftpk.fftfreq(len(FFT), 1.0/s_rate)[range(len(FFT_)//2)]
    sorted = np.sort(FFT)[::-1]
    ma1, ma2 , ma3 = sorted[[0,1,2]]
    max_amp_index = [np.where(FFT == sorted[0])[0], np.where(FFT == sorted[1])[0], np.where(FFT == sorted[2])[0]]
    f1, f2, f3 = np.concatenate(freqs[max_amp_index])

    # Storing the features in a list
    features = [rms,mean,ma1,ma2,ma3,f1,f2,f3]

    # Creating a dataframe
    input_data = pd.DataFrame([features])
    input_data.columns = ["RMS","Mean","MA1","MA2","MA3","F1","F2","F3"]
    arr = input_data.to_dict('records')
    print(arr)
    # Making a POST request to the API
    response = requests.post("http://127.0.0.1:8000/predict",json=input_data.to_dict('records')[0])
    print(response.json())
    print(response.json()['prediction'])
    prediction = response.json()
    return prediction['prediction']
    

def main():
    import streamlit as st
    import time

    # Creating Streamlit buttons
    record_button = st.button('Record', key='record_button')
    stop_button = st.button('Stop', key='stop_button')

    # Creating Streamlit text output
    output = st.empty()

    # Flag to control recording state
    is_rec = True

    # This function records the data and shows the predictions
    def start_recording():
        global is_rec
        while is_rec:
            output.text('Recording...')
            new_preds = predict()
            output.text(new_preds)
            if new_preds == 1.0:
                st.markdown("**Machine ID: Off condition with noise**")
                st.image("D:\\Documents\\SLIoT\\ML App\\ui\\off.png", caption="Off condition with noise", width=200)
                send_error_request("Off condition")
            elif new_preds == 2.0:
                st.markdown("**Machine ID: Healthy condition**")
                st.image("D:\\Documents\\SLIoT\\ML App\\ui\\good_.png", caption="Healthy condition", width=200)
                send_error_request("Healthy condition")
            elif new_preds == 3.0:
                st.markdown("**Machine ID: Bearing Fault**")
                st.image("D:\\Documents\\SLIoT\\ML App\\ui\\mfault1_.png", caption="Bearing Fault", width=200)
                send_error_request("Bearing Fault")
            else:
                st.markdown("**Machine ID: Fan Fault**")
                st.image("D:\\Documents\\SLIoT\\ML App\\ui\\mfault2.png", caption="Fan Fault", width=200)
                send_error_request("Fan Fault")
            # st.write(predictions)
            time.sleep(5)

    # Function to stop recording
    def stop_recording():
        global is_rec
        is_rec = False
        output.text('Recording stopped.')

    # Handling button clicks
    if record_button:
        start_recording()

    if stop_button:
        stop_recording()

    # Creating threads
    x= threading.Thread(target=start_recording)
    y= threading.Thread(target=stop_recording)
    x.start()
    y.start()
    x.join()
    y.join()


if __name__ == "__main__":
    main()