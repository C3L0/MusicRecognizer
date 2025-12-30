# tests/test_app.py
import pytest
import numpy as np
import io
import soundfile as sf
from fastapi.testclient import TestClient
from src.app import app  # Importing your FastAPI instance

client = TestClient(app)

def test_read_health():
    """Test the /health endpoint used for system monitoring."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_invalid_format():
    """Verify that the API rejects non-audio files (e.g., text files)."""
    files = {"file": ("test.txt", b"hello world", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    # Matches the English detail string from app.py
    assert "File must be an audio format" in response.json()["detail"]

def test_predict_no_file():
    """Verify that the API returns an error if no file is sent."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity (FastAPI standard)

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_predict_success():
    """Integration test: Verify the model processes a generated sound (white noise)."""
    # Generate 1 second of silence at 16kHz
    data = np.zeros(16000)
    byte_io = io.BytesIO()
    sf.write(byte_io, data, 16000, format='mp3')
    byte_io.seek(0)

    files = {"file": ("silence.mp3", byte_io, "audio/mp3")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "filename" in response.json()

def test_predict_music():
    """Real-world test: Verify the model correctly identifies a music sample."""
    file_path = "tests/samples/piano.mp3"

    with open(file_path, "rb") as f:
        files = {"file": ("piano.mp3", f, "audio/mp3")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    prediction = response.json()["prediction"]
    
    # Debug print for console inspection if the test fails
    print(f"DEBUG Prediction: {prediction}")

    # Handling different response formats (list or single string)
    if isinstance(prediction, list):
        assert any("music" in p.lower() for p in prediction)
    else:
        assert "music" in prediction.lower()