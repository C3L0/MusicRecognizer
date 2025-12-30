import os
import sys
import logging
import psutil
import torch
import librosa
import uvicorn
import gradio as gr
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from transformers import AutoProcessor, ASTForAudioClassification

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- APP INITIALIZATION ---
load_dotenv()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

app = FastAPI(title="Sound Recognizer API")

# --- MODEL LOADING ---
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
logger.info(f"Loading model: {MODEL_ID}")

# use_fast=True removes the UserWarning
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = ASTForAudioClassification.from_pretrained(MODEL_ID)

# --- CORE LOGIC ---
def process_audio(audio_path):
    """Shared inference logic for both API and UI."""
    if audio_path is None:
        return None
    
    try:
        # Load and resample to 16kHz
        y, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(y, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Calculate probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        top5_prob, top5_indices = torch.topk(probs, 5)
        
        return {
            model.config.id2label[idx.item()]: float(prob) 
            for prob, idx in zip(top5_prob, top5_indices)
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {"error": str(e)}

# --- FASTAPI ROUTES ---
@app.get("/health")
def health():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "model": MODEL_ID}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    """REST API endpoint for sound classification."""
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio format.")
    
    logger.info(f"API Request: {file.filename}")
    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        full_res = process_audio(temp_path)
        if "error" in full_res:
            raise HTTPException(status_code=500, detail=full_res["error"])
            
        # Return the top prediction for the simple API response
        best_prediction = list(full_res.keys())[0]
        return {"prediction": best_prediction, "filename": file.filename}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/admin/stats")
async def get_admin_stats(token: str = None):
    """Secure endpoint for system resource monitoring."""
    if token != ADMIN_TOKEN:
        logger.warning("Unauthorized admin access attempt.")
        raise HTTPException(status_code=401, detail="Unauthorized")
        
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

# --- GRADIO INTERFACE ---
custom_css = "#title { text-align: center; color: #1a73e8; }"

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Music & Sound Recognizer", elem_id="title")
    gr.Markdown("Analyze audio using the Audio Spectrogram Transformer (AST).")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Record or Upload Audio", 
                type="filepath",
                sources=["microphone", "upload"],
                waveform_options=gr.WaveformOptions(
                    waveform_color="#2196F3",
                    waveform_progress_color="#BBDEFB",
                )
            )
            submit_btn = gr.Button("Analyze Now", variant="primary")

        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Predictions")
        
    with gr.Accordion("🛠 Admin Panel", open=False):
        admin_key = gr.Textbox(label="Admin Key", type="password")
        admin_output = gr.JSON(label="System Status")
        btn_stats = gr.Button("Refresh Stats")
        
        def show_stats(key):
            if key == ADMIN_TOKEN:
                mem = psutil.virtual_memory()
                return {
                    "cpu_percent": psutil.cpu_percent(), 
                    "ram_percent": mem.percent,
                    "ram_used_mb": mem.used // (1024**2)
                }
            return {"error": "Invalid Key"}

        btn_stats.click(fn=show_stats, inputs=admin_key, outputs=admin_output)

    # Trigger analysis on button click or audio change
    submit_btn.click(fn=process_audio, inputs=audio_input, outputs=label_output)
    audio_input.change(fn=process_audio, inputs=audio_input, outputs=label_output)

# --- MOUNTING ---
# Mount Gradio to the root of FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    logger.info("Starting local server...")
    uvicorn.run(app, host="127.0.0.1", port=7860)