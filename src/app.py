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

# 1. Mise à jour de la fonction de traitement pour renvoyer des probabilités
def process_audio(audio_path):
    if audio_path is None: return None
    
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Calcul des probabilités pour un affichage graphique (Barres)
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top5_prob, top5_indices = torch.topk(probs, 5)
    
    # On crée un dictionnaire {Label: Score}
    confidences = {
        model.config.id2label[idx.item()]: float(prob) 
        for prob, idx in zip(top5_prob, top5_indices)
    }
    return confidences

# 2. Création de l'interface graphique avec gr.Blocks

custom_css = """
.gradio-container { background-color: #f0f2f5; }
#title { text-align: center; color: #1a73e8; }
"""
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown(
        """
        # 🎵 Music & Sound Recognizer
        ### Analyse intelligente basée sur le modèle AST (Audio Spectrogram Transformer)
        """
    )
    
    with gr.Row():
        # Colonne de gauche : Input
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Fichier Audio", 
                type="filepath",
                waveform_options=gr.WaveformOptions(
                    wave_color="#2196F3",
                    pending_color="#BBDEFB",
                )
            )
            submit_btn = gr.Button("Analyser le son", variant="primary")

        # Colonne de droite : Résultats graphiques
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=5, label="Prédictions")

    # Section exemples pour faciliter le test
    gr.Examples(
        examples=["tests/samples/piano.mp3"],
        inputs=audio_input
    )

    # Logique du bouton
    submit_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=label_output
    )

# 3. Fusion avec FastAPI
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")