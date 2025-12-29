# tests/test_app.py
import pytest
from fastapi.testclient import TestClient
from src.app import app  # Import de ton instance FastAPI

client = TestClient(app)

def test_read_health():
    """On teste /health au lieu de / car / est maintenant l'interface Gradio."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_invalid_format():
    """Vérifie que l'API rejette les fichiers qui ne sont pas audio."""
    files = {"file": ("test.txt", b"hello world", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    # On met le texte EXACT que tu as écrit dans app.py
    assert "Le fichier doit être un audio" in response.json()["detail"]

def test_predict_no_file():
    """Vérifie que l'API renvoie une erreur si aucun fichier n'est envoyé."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity (FastAPI standard)

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_predict_success():
    """Test d'intégration : vérifie que le modèle traite un son (bruit blanc)."""
    # Générer un petit fichier WAV vide (bruit blanc) en mémoire pour le test
    import numpy as np
    import io
    import soundfile as sf

    # 1 seconde de silence à 16kHz
    data = np.zeros(16000)
    byte_io = io.BytesIO()
    sf.write(byte_io, data, 16000, format='WAV')
    byte_io.seek(0)

    files = {"file": ("silence.wav", byte_io, "audio/mp3")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "filename" in response.json()

def test_predict_music():
    file_path = "tests/samples/piano.mp3"

    with open(file_path, "rb") as f:
        files = {"file": ("piano.mp3", f, "audio/mp3")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    prediction = response.json()["prediction"]
    
    # Debug pour voir ce qu'on reçoit exactement dans la console si ça échoue
    print(f"DEBUG Prediction: {prediction}")

    # Si c'est une liste
    if isinstance(prediction, list):
        assert any("music" in p.lower() for p in prediction)
    # Si c'est une simple chaîne
    else:
        assert "music" in prediction.lower()