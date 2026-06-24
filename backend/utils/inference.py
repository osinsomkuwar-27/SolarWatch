# backend/utils/inference.py
import tensorflow as tf
import numpy as np
import pickle
import os

# Load everything once when server starts
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, '..', 'models')

cnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'cnn_model.h5'))
lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'))

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

CLASSES = ['A', 'B', 'C', 'M', 'X']
WINDOW = 60
FEATURES = ['log_flux', 'd_flux', 'rolling_max_5m', 'variance_10m', 'rolling_mean_10m']

def run_inference(recent_docs):
    """
    recent_docs: list of last 60 MongoDB documents, oldest first
    each doc must have: log_flux, d_flux, rolling_max_5m, variance_10m, rolling_mean_10m
    returns: dict with class probs and alert probability
    """
    if len(recent_docs) < WINDOW:
        return None

    X = np.array([[doc[f] for f in FEATURES] for doc in recent_docs[-WINDOW:]])
    X_scaled = scaler.transform(X).reshape(1, WINDOW, 5)

    cnn_probs = cnn_model.predict(X_scaled, verbose=0)[0].tolist()
    lstm_prob = float(lstm_model.predict(X_scaled, verbose=0)[0][0])

    return {
        "predicted_class": CLASSES[int(np.argmax(cnn_probs))],
        "cnn_class_probs": cnn_probs,   # [A_prob, B_prob, C_prob, M_prob, X_prob]
        "lstm_alert_prob": lstm_prob     # probability of M+ in next hour
    }