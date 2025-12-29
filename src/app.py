import io
import torch
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoFeatureExtractor, ASTForAudioClassification

app = FastAPI(title="Music Recognition API")

# 1. Configuration du modèle Hugging Face
# Ce modèle est entraîné sur AudioSet (527 classes de sons/musiques)
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

print("Chargement du modèle...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = ASTForAudioClassification.from_pretrained(MODEL_NAME)
print("Modèle prêt !")

def process_audio(audio_bytes):
    # Charger l'audio depuis les bytes (échantillonnage à 16kHz requis par AST)
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    
    # Préparer les inputs pour le modèle
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Récupérer la classe avec le score le plus haut
    predicted_class_ids = torch.argmax(logits, dim=-1).item()
    prediction_label = model.config.id2label[predicted_class_ids]
    
    return prediction_label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Vérification sommaire du format
    if not file.filename.endswith(('.mp3', '.wav', '.flac')):
        raise HTTPException(status_code=400, detail="Format de fichier non supporté.")

    try:
        audio_content = await file.read()
        label = process_audio(audio_content)
        return {"filename": file.filename, "prediction": label}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "API de reconnaissance musicale active. Utilisez /predict pour envoyer un fichier."}