import os
import numpy as np
import wfdb
import pywt
import joblib
import streamlit as st
from tensorflow.keras.models import load_model
from tempfile import TemporaryDirectory

# Load your trained components
model = load_model("ecg_multiclass_lstm_model.h5")
scaler = joblib.load("ecg_scaler.save")
label_encoder = joblib.load("label_encoder.save")

TARGET_LENGTH = 9000

# Feature extraction
def process_signal(signal, target_length=9000):
    if len(signal) > target_length:
        return signal[:target_length]
    elif len(signal) < target_length:
        return np.pad(signal, (0, target_length - len(signal)), mode='constant')
    return signal

def wavelet_decomposition(signal, level=1):
    coeffs = pywt.wavedec(signal, 'db4', level=level)
    return np.hstack(coeffs)

def RR_features(signal):
    return np.array([np.mean(signal), np.std(signal)])

def spectral_features(signal):
    return np.fft.fft(signal).real[:10]

def extract_features(signal):
    return np.hstack([
        wavelet_decomposition(signal),
        RR_features(signal),
        spectral_features(signal)
    ])

# Streamlit App UI
st.title("📈 ECG Classification from .hea/.dat Upload")
st.write("Upload a PhysioNet ECG record (.hea and .dat) to classify the ECG signal.")

uploaded_files = st.file_uploader("Upload both .hea and .dat files", type=["hea", "dat"], accept_multiple_files=True)

if uploaded_files:
    with TemporaryDirectory() as tmpdir:
        filenames = []
        for file in uploaded_files:
            path = os.path.join(tmpdir, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            filenames.append(path)

        # Find common record name
        record_names = list(set(os.path.splitext(os.path.basename(f))[0] for f in filenames))
        if len(record_names) != 1:
            st.error("Please upload matching .hea and .dat files for the same record.")
        else:
            record_path = os.path.join(tmpdir, record_names[0])
            try:
                record = wfdb.rdrecord(record_path)
                signal_matrix = record.p_signal
                ecg_signal = signal_matrix[:, 0].flatten()

                # Process and predict
                ecg_signal = process_signal(ecg_signal, TARGET_LENGTH)
                features = extract_features(ecg_signal)
                features_scaled = scaler.transform([features])
                features_reshaped = features_scaled.reshape((1, features_scaled.shape[1], 1))
                prediction_probs = model.predict(features_reshaped)[0]
                predicted_class_index = np.argmax(prediction_probs)
                predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

                # Display result
                st.success(f"✅ Predicted Class: **{predicted_label}**")
                st.subheader("📊 Class Probabilities")
                for cls, prob in zip(label_encoder.classes_, prediction_probs):
                    st.write(f"- {cls}: **{prob:.3f}**")

            except Exception as e:
                st.error(f"❌ Error reading files: {e}")
