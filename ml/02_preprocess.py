# ml/02_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv("ml/data/raw_data.csv")
df['time'] = pd.to_datetime(df['time_tag'])
df = df.sort_values('time').reset_index(drop=True)

# Remove bad/zero flux values
df = df[df['flux'] > 0]

# Feature engineering — these are the 5 things your model sees
df['log_flux'] = np.log10(df['flux'])
df['d_flux'] = df['flux'].diff().fillna(0)           # rate of change
df['rolling_max_5m'] = df['flux'].rolling(5).max()   # max in last 5 min
df['variance_10m'] = df['flux'].rolling(10).var()    # how chaotic last 10 min
df['rolling_mean_10m'] = df['flux'].rolling(10).mean()

# Label each minute with flare class number
def label(flux):
    if flux >= 1e-4: return 4    # X
    elif flux >= 1e-5: return 3  # M
    elif flux >= 1e-6: return 2  # C
    elif flux >= 1e-7: return 1  # B
    else: return 0               # A

df['class_label'] = df['flux'].apply(label)
df.dropna(inplace=True)
df.to_csv("ml/data/processed_data.csv", index=False)

print("Class distribution (how many minutes of each class):")
print(df['class_label'].value_counts().sort_index())
# You'll see A dominates — that's normal. Flares are rare.

# Add this to 02_preprocess.py
WINDOW = 60  # look back 60 minutes
FEATURES = ['log_flux', 'd_flux', 'rolling_max_5m', 'variance_10m', 'rolling_mean_10m']

X, y = [], []
for i in range(WINDOW, len(df)):
    X.append(df[FEATURES].iloc[i-WINDOW:i].values)
    y.append(df['class_label'].iloc[i])

X = np.array(X)  # shape: (samples, 60, 5)
y = np.array(y)  # shape: (samples,)

# Scale
scaler = StandardScaler()
X_flat = X.reshape(-1, 5)
X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

with open("ml/saved_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split
split = int(0.8 * len(X))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Save for use in training scripts
np.save("ml/data/X_train.npy", X_train)
np.save("ml/data/X_test.npy", X_test)
np.save("ml/data/y_train.npy", y_train)
np.save("ml/data/y_test.npy", y_test)