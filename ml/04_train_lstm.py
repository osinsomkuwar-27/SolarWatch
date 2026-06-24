# ml/04_train_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

X_train = np.load("ml/data/X_train.npy")
y_train = np.load("ml/data/y_train.npy")

# Binary: 1 if M or X class (dangerous), 0 if A/B/C (safe)
y_binary = (y_train >= 3).astype(int)

model = models.Sequential([
    layers.Input(shape=(60, 5)),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Flares are rare so we weight them heavily
class_weights = {0: 1, 1: 15}

early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_binary,
    validation_split=0.1,
    epochs=30,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

model.save("ml/saved_models/lstm_model.h5")
print("LSTM saved to ml/saved_models/lstm_model.h5")