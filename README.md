---
title: Sounds Recognizer
emoji: 📉
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
short_description: 'School project '
---

# Sound Recognizer

A MLOps project for audio classification using the **Audio Spectrogram Transformer (AST)**. This application provides both a user-friendly web interface and a robust REST API.

https://huggingface.co/spaces/c3loo/sounds_recognizer

## Features

- **Hybrid Architecture**: Built with FastAPI for the backend and Gradio for the frontend.
- **Audio Classification**: Recognizes over 500 sound classes (Music, Instruments, Environmental sounds, etc.).
- **Live Input**: Support for file uploads and direct microphone recording.
- **Monitoring**: Integrated Admin panel and health-check endpoints for resource tracking.

## Tech Stack

- **Model**: `MIT/ast-finetuned-audioset-10-10-0.4593` (Transformers)
- **Backend**: FastAPI
- **Frontend**: Gradio 6.0
- **Environment Management**: `uv`
- **CI/CD**: GitHub Actions

## How to Use

### Web Interface
Simply open the Space, upload an audio file or click the microphone icon to record. The model will display the Top 5 most likely sound categories.

### API Access
You can interact with the model programmatically:

```bash
curl -X 'POST' \
  '[https://your-space-url.hf.space/predict](https://your-space-url.hf.space/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_audio_file.mp3;type=audio/mpeg'
```
**You can find more details in the "API" tab of the Space.**

## Development & Testing
This project follows strict CI/CD practices. Every push to main triggers:

Unit Testing: Automated tests using pytest.

Clean Deployment: Automated filtering to remove test samples and sensitive config before pushing to production.

To run tests locally:
```bash

uv run pytest
```

## Admin Access
System metrics are available at /admin/stats?token=YOUR_TOKEN or via the hidden Admin Panel in the UI (requires ADMIN_TOKEN).
