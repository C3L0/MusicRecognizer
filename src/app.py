import gradio as gr
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, ASTForAudioClassification

# 1. Initialisation de FastAPI
app = FastAPI(title="Music Recognizer API")

# 2. Chargement du modèle (une seule fois pour les deux interfaces)
model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
processor = AutoProcessor.from_pretrained(model_id)
model = ASTForAudioClassification.from_pretrained(model_id)

def process_audio(audio_path):
    """Fonction noyau pour la prédiction (partagée)"""
    # Charger et rééchantillonner à 16kHz (requis par AST)
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Préparation des inputs
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Récupérer le label avec le score le plus haut
    predicted_class_ids = torch.argmax(logits, dim=-1).item()
    prediction = model.config.id2label[predicted_class_ids]
    return prediction

# --- PARTIE API (FastAPI) ---
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un audio.")
    
    # Sauvegarde temporaire pour traitement
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        res = process_audio(temp_path)
        return {"prediction": res, "filename": file.filename}
    finally:
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
def health():
    return {"status": "ok"}

# --- PARTIE INTERFACE (Gradio) ---
def gradio_interface(audio):
    if audio is None:
        return "Veuillez uploader un fichier."
    return process_audio(audio)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Audio(type="filepath", label="Déposez votre musique ici"),
    outputs=gr.Label(label="Résultat de la détection"),
    title="🎵 Music Genre Recognizer",
    description="Uploadez un extrait sonore et laissez l'IA (AST Model) identifier le contenu.",
    examples=["tests/samples/piano.mp3"] # Si tu as gardé le fichier
)

# --- FUSION ---
# On monte l'interface Gradio sur la racine "/" de FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

# import io
# import torch
# import librosa
# import numpy as np
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from transformers import AutoFeatureExtractor, ASTForAudioClassification
# 
# app = FastAPI(title="Music Recognition API")
# 
# # 1. Configuration du modèle Hugging Face
# # Ce modèle est entraîné sur AudioSet (527 classes de sons/musiques)
# MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
# 
# print("Chargement du modèle...")
# feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
# model = ASTForAudioClassification.from_pretrained(MODEL_NAME)
# print("Modèle prêt !")
# 
# def process_audio(audio_bytes):
#     # Charger l'audio depuis les bytes (échantillonnage à 16kHz requis par AST)
#     audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
#     
#     # Préparer les inputs pour le modèle
#     inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
#     
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         
#     # Récupérer la classe avec le score le plus haut
#     predicted_class_ids = torch.argmax(logits, dim=-1).item()
#     prediction_label = model.config.id2label[predicted_class_ids]
#     
#     return prediction_label
# 
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Vérification sommaire du format
#     if not file.filename.endswith(('.mp3', '.wav', '.flac')):
#         raise HTTPException(status_code=400, detail="Format de fichier non supporté.")
# 
#     try:
#         audio_content = await file.read()
#         label = process_audio(audio_content)
#         return {"filename": file.filename, "prediction": label}
#     except Exception as e:
#         return {"error": str(e)}
# 
# @app.get("/")
# def home():
#     return {"message": "API de reconnaissance musicale active. Utilisez /predict pour envoyer un fichier."}