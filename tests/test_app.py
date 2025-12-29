# tests/test_app.py
import pytest
from fastapi.testclient import TestClient
from src.app import app  # Import de ton instance FastAPI

client = TestClient(app)

def test_read_main():
    """Vérifie que la route d'accueil répond bien."""
    response = client.get("/")
    assert response.status_code == 200
    assert "API de reconnaissance musicale active" in response.json()["message"]

def test_predict_no_file():
    """Vérifie que l'API renvoie une erreur si aucun fichier n'est envoyé."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity (FastAPI standard)

def test_predict_invalid_format():
    """Vérifie que l'API rejette les fichiers qui ne sont pas audio."""
    # Création d'un faux fichier texte
    files = {"file": ("test.txt", b"hello world", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Format de fichier non supporté" in response.json()["detail"]

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

    files = {"file": ("silence.wav", byte_io, "audio/wav")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "filename" in response.json()

    def test_predict_piano():
    """Vérifie que le modèle identifie correctement un piano."""
    file_path = "tests/samples/piano.wav"
    
    with open(file_path, "rb") as f:
        files = {"file": ("piano.wav", f, "audio/wav")}
        response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    
    # On vérifie si le mot 'piano' est dans le top des prédictions
    # (Selon les labels du modèle AST)
    assert any("piano" in p.lower() for p in prediction)