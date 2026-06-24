# ml/01_download_data.py
import requests
import pandas as pd

# This gives you 7 days — for 3 years you need the NGDC archive above
url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
data = requests.get(url).json()

# Filter only the 0.1-0.8nm channel (standard flare channel)
filtered = [d for d in data if d.get('energy') == '0.1-0.8nm']

df = pd.DataFrame(filtered)
df.to_csv("ml/data/raw_data.csv", index=False)
print(f"Saved {len(df)} rows")
print(df.head())