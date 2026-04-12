import requests
import json

# Get a single prediction from the server
response = requests.post("http://localhost:8000/predict", json={
    "context": "Test context",
    "utterance": "Test utterance",
    "include_analysis": False
})

pred = response.json()
print("Server prediction:")
print(json.dumps(pred, indent=2))
print("\nPrediction type:", type(pred['predictions']))
print("First value:", pred['predictions']['Non-Judgmental Language'], type(pred['predictions']['Non-Judgmental Language']))
