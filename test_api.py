import requests

url = "http://localhost:5000/predict"
headers = {
    "Content-Type": "application/json"
}
data = {
    "text": "I can't stand that bitch, she's always talking trash."
}

response = requests.post(url, json=data, headers=headers)
print(response.json())