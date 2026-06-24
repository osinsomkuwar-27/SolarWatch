# ml/05_evaluate.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X_test = np.load("ml/data/X_test.npy")
y_test = np.load("ml/data/y_test.npy")

# CNN evaluation
cnn = tf.keras.models.load_model("ml/saved_models/cnn_model.h5")
y_pred = np.argmax(cnn.predict(X_test), axis=1)

print("=== CNN RESULTS ===")
print(classification_report(y_test, y_pred, target_names=['A','B','C','M','X']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['A','B','C','M','X'],
            yticklabels=['A','B','C','M','X'])
plt.title('CNN Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("ml/saved_models/confusion_matrix.png")
print("Confusion matrix saved")

# LSTM evaluation
lstm = tf.keras.models.load_model("ml/saved_models/lstm_model.h5")
y_binary_test = (y_test >= 3).astype(int)
y_pred_binary = (lstm.predict(X_test) > 0.3).astype(int).flatten()

print("\n=== LSTM ALERT MODEL RESULTS ===")
print(classification_report(y_binary_test, y_pred_binary,
      target_names=['Safe','M+ Alert']))