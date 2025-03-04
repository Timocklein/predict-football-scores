import pandas as pd
import json
import requests

# Laad de holdout data in
X_holdout = pd.read_csv('X_holdout.csv')  

# Selecteer een willekeurige sample uit de holdout data
sample = X_holdout.sample(n=1).iloc[0].to_dict()

# Converteer de sample naar een JSON object
payload = [sample]
print(payload)

# Maak een POST request naar de API
url = "http://127.0.0.1:8000/predict"
response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())