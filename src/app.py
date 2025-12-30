import gradio as gr
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import librosa
import numpy as np
import os
from transformers import AutoProcessor, ASTForAudioClassification

# 1. Initialisation de FastAPI
app = FastAPI(title="Music Recognizer API")

# 2. Chargement du modèle (Ajout de use_fast=True pour enlever le warning)
model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = ASTForAudioClassification.from_pretrained(model_id)

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

# --- CONFIGURATION GRADIO ---

custom_css = "#title { text-align: center; color: #1a73e8; }"

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎵 Music & Sound Recognizer", elem_id="title")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Fichier Audio", 
                type="filepath",
                # CORRECTION : Noms des arguments pour Gradio 6.0
                waveform_options=gr.WaveformOptions(
                    waveform_color="#2196F3",
                    waveform_progress_color="#BBDEFB",
                )
            )
            submit_btn = gr.Button("Analyser le son", variant="primary")

        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Prédictions")

    submit_btn.click(fn=process_audio, inputs=audio_input, outputs=label_output)

# MONTAGE FINAL (Gradio vient s'ajouter à FastAPI sans écraser les routes)
app = gr.mount_gradio_app(app, demo, path="/")