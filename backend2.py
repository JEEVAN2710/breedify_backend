# backend.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pathlib import Path
import io

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

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_cattle_model.pth"

state_dict = torch.load(MODEL_PATH, map_location=device)
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
# Classification
# =======================================================

def predict(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        confidence = probs[0, predicted_class].item()

    if confidence < threshold:
        return {"label": "Unknown", "confidence": round(confidence, 3)}

    return {"label": class_names[predicted_class.item()], "confidence": round(confidence, 3)}

@app.get("/")
def root():
    return {"message": "Backend is running 🚀"}

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)
    return result
