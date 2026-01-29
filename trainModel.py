import numpy as np
import scipy.io as sio
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import pywt
import matplotlib.pyplot as plt
import joblib # ✅ Added for saving scaler and label encoder

# Paths
data_folder = "../sample2017/validation"   
csv_file = "../sample2017/validation/REFERENCE.csv"  

# Load label CSV
tb = pd.read_csv(csv_file, header=None, names=['Filename', 'Label'])

Signals = []
Labels = []

# Load signals and corresponding labels
for index, row in tb.iterrows():
    file_path = os.path.join(data_folder, f"{row['Filename']}.mat")
    if os.path.exists(file_path):
        fileData = sio.loadmat(file_path)
        Signals.append(fileData['val'].flatten())
        Labels.append(row['Label'])

Labels = np.array(Labels)

# Padding/Truncating
target_length = 9000  
def process_signal(signal, target_length=9000):
    if len(signal) > target_length:
        return signal[:target_length] # Truncate
    elif len(signal) < target_length:
        return np.pad(signal, (0, target_length - len(signal)), mode='constant') # Pad
    return signal

# Class-wise storage
class_map = {'A': [], 'N': [], 'O': []}
for signal, label in zip(Signals, Labels):
    if label in class_map:
        processed_signal = process_signal(signal, target_length)
        class_map[label].append(processed_signal)

afibX = np.array(class_map['A'])
normalX = np.array(class_map['N'])
OtX = np.array(class_map['O'])

afibY = np.full(len(afibX), 'A')
normalY = np.full(len(normalX), 'N')
OtY = np.full(len(OtX), 'O')

# Train-test split
X_train_A, X_test_A, Y_train_A, Y_test_A = train_test_split(afibX, afibY, test_size=0.2, random_state=42)
X_train_N, X_test_N, Y_train_N, Y_test_N = train_test_split(normalX, normalY, test_size=0.2, random_state=42)
X_train_O, X_test_O, Y_train_O, Y_test_O = train_test_split(OtX, OtY, test_size=0.2, random_state=42)

X_train = np.concatenate([X_train_A, X_train_N, X_train_O])
Y_train = np.concatenate([Y_train_A, Y_train_N, Y_train_O])
X_test = np.concatenate([X_test_A, X_test_N, X_test_O])
Y_test = np.concatenate([Y_test_A, Y_test_N, Y_test_O])

# Feature extraction
def wavelet_decomposition(signal, level=1):
    coeffs = pywt.wavedec(signal, 'db4', level=level)
    return np.hstack(coeffs)

def RR_features(signal):
    return np.array([np.mean(signal), np.std(signal)])

def spectral_features(signal):
    return np.fft.fft(signal).real[:10]

X_train_features = np.array([np.hstack((wavelet_decomposition(x), RR_features(x), spectral_features(x))) for x in X_train])
X_test_features = np.array([np.hstack((wavelet_decomposition(x), RR_features(x), spectral_features(x))) for x in X_test])

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Encode labels
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

# LSTM model
def lstm_model(X_train, Y_train, X_test, Y_test, epochs=50, batch_size=64):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, Y_test), callbacks=[early_stopping])
    
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
    return model

# Reshape for LSTM
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Train and save model
model = lstm_model(X_train_scaled, Y_train_encoded, X_test_scaled, Y_test_encoded, epochs=5, batch_size=64)

# ✅ Save the model and preprocessing objects
model.save("ecg_multiclass_lstm_model.h5") # Save LSTM model
joblib.dump(scaler, "ecg_scaler.save") # Save StandardScaler
joblib.dump(label_encoder, "label_encoder.save") # Save LabelEncoder

print("✅ Model, scaler, and label encoder saved successfully.")
