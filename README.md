# Whisper app

A simple GUI wrapper around OpenAI's Whisper and PyAnnote to transcribe and diarize speech audio files.

## Features

- Select audio files and model version
- Choose output format (`.txt` or `.docx`) and location
- Automatic speaker diarization with PyAnnote
- Specify audio language
- Simple GUI via PySimpleGUI

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

Requires a valid Hugging Face token in `config.ini`:

```
[huggingface]
token = YOUR_HF_TOKEN
```

## Project structure

```
config.ini
src/
	├── __init__.py
	├── __main__.py
	├── app.py               # GUI and control flow
	├── transcription.py     # Whisper and diarization pipeline
	├── util.py              # Validation and file-saving helpers
tests/
	├── __init__.py
	├── test_app.py
	├── test_util.py
	├── test_transcription.py
```

## Usage

```
python -m src
```
