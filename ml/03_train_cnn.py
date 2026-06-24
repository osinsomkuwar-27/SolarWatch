# ml/03_train_cnn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

X_train = np.load("ml/data/X_train.npy")
y_train = np.load("ml/data/y_train.npy")

model = models.Sequential([
    layers.Input(shape=(60, 5)),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# M and X class get higher weight because they're rare but important
class_weights = {0: 1, 1: 2, 2: 5, 3: 20, 4: 50}

early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

model.save("ml/saved_models/cnn_model.h5")
print("CNN saved to ml/saved_models/cnn_model.h5")