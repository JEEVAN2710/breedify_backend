# backend.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import cv2
import numpy as np

# =======================================================
# Config
# =======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 41
class_names = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni',
    'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam',
    'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 'Malnad_gidda',
    'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
    'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery','Vechur', "Unknown"
]
threshold = 0.275

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =======================================================
# Load Model
# =======================================================
model = models.efficientnet_v2_s(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

state_dict = torch.load("best_cattle_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# =======================================================
# FastAPI App
# =======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# Detection + Classification
# =======================================================
def detect_and_crop(image_bytes: bytes):
    """Detect cow region using OpenCV (Haar Cascade) and crop."""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Load pretrained cow detector (replace with YOLO/other model if needed)
    cow_cascade = cv2.CascadeClassifier("haarcascade_cow.xml")  # <-- you need this XML file

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cows = cow_cascade.detectMultiScale(gray, 1.1, 4)

    if len(cows) == 0:
        return None  # no cow detected

    # Take the largest detected cow region
    x, y, w, h = max(cows, key=lambda b: b[2] * b[3])
    cropped = img[y:y+h, x:x+w]
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return cropped_pil


def predict(image_bytes: bytes):
    cropped_img = detect_and_crop(image_bytes)

    if cropped_img is None:
        return {"label": "No cow detected", "confidence": 0.0}

    img_t = transform(cropped_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        confidence = probs[0, predicted_class].item()

    if confidence < threshold:
        return {"label": "Unknown", "confidence": round(confidence, 3)}

    return {"label": class_names[predicted_class.item()], "confidence": round(confidence, 3)}


@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)
    return result
