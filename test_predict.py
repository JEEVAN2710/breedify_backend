# e:\backup\breedify_backend\test_predict.py
import requests

with open("download.jpeg", "rb") as image_file:
    response = requests.post(
        "http://127.0.0.1:8000/predict/",
        files={"file": image_file},  # FastAPI expects 'file'
    )

print(response.status_code)
print(response.json())
