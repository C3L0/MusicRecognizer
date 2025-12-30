import logging
import gradio as gr
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import librosa
import numpy as np
import os
from transformers import AutoProcessor, ASTForAudioClassification
import psutil
from dotenv import load_dotenv

# 1. Initialisation de FastAPI
app = FastAPI(title="Sound Recognizer API")

# 2. Chargement du modèle (Ajout de use_fast=True pour enlever le warning)
model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = ASTForAudioClassification.from_pretrained(model_id)

load_dotenv()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

def get_system_info():
    # Récupère l'utilisation de la RAM
    mem = psutil.virtual_memory()
    # Récupère l'utilisation du CPU
    cpu = psutil.cpu_percent()
    return f"💻 CPU: {cpu}% | RAM: {mem.percent}% ({mem.used // (1024**2)}MB / {mem.total // (1024**2)}MB)"

def process_audio(audio_path):
    """Logique partagée entre l'API et l'Interface"""
    if audio_path is None: return None
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top5_prob, top5_indices = torch.topk(probs, 5)
    return {model.config.id2label[idx.item()]: float(prob) for prob, idx in zip(top5_prob, top5_indices)}

# --- ÉTAPE CRUCIALE : DÉFINIR LES ROUTES AVANT LE MOUNT ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un audio.")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # On ne récupère que le premier résultat pour l'API simple
        full_res = process_audio(temp_path)
        best_prediction = list(full_res.keys())[0]
        return {"prediction": best_prediction, "filename": file.filename}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/admin/stats")
async def get_admin_stats(token: str = None):
    # Optionnel : une petite sécurité par token
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Non autorisé")
        
    mem = psutil.virtual_memory()
    return {
        "cpu_usage_percent": psutil.cpu_percent(),
        "ram_usage": {
            "percent": mem.percent,
            "used_mb": mem.used // (1024**2),
            "total_mb": mem.total // (1024**2)
        },
        "active_pid": os.getpid()
    }

# --- CONFIGURATION GRADIO ---

custom_css = "#title { text-align: center; color: #1a73e8; }"

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎵 Music & Sound Recognizer", elem_id="title")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Enregistrez ou déposez un son", 
                type="filepath",
                # On active explicitement le micro et l'upload de fichier
                sources=["microphone", "upload"],
                waveform_options=gr.WaveformOptions(
                    waveform_color="#2196F3",
                    waveform_progress_color="#BBDEFB",
                )
            )
            submit_btn = gr.Button("🚀 Analyser maintenant", variant="primary")

        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Prédictions")
        
        with gr.Accordion("🛠 Panneau Admin", open=False): # Fermé par défaut
            admin_key = gr.Textbox(label="Clé Admin", type="password")
            admin_output = gr.JSON(label="État du Système")
            btn_stats = gr.Button("Actualiser les stats")
            
            def show_stats(key):
                if key == ADMIN_TOKEN: # Ton code secret
                    mem = psutil.virtual_memory()
                    return {"cpu": psutil.cpu_percent(), "ram_percent": mem.percent}
                return {"error": "Clé invalide"}

            btn_stats.click(fn=show_stats, inputs=admin_key, outputs=admin_output)

    submit_btn.click(fn=process_audio, inputs=audio_input, outputs=label_output)
    
    # ANALYSE AUTOMATIQUE : se déclenche au changement (upload ou fin d'enregistrement)
    audio_input.change(fn=process_audio, inputs=audio_input, outputs=label_output)

# MONTAGE FINAL (Gradio vient s'ajouter à FastAPI sans écraser les routes)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    print(ADMIN_TOKEN)
    uvicorn.run(app, host="127.0.0.1", port=7860)