import os
import configparser

from docx import Document
import PySimpleGUI as sg
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment


def load_huggingface_token(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    huggingface_token = config["huggingface"]["token"]
    return huggingface_token

def create_layout():
    layout = [
        [sg.Text("Select an audio file:")],
        [sg.Input(), sg.FileBrowse(key="AUDIO_FILE", file_types=(("Audio Files", "*.wav *.mp3 *.flac"),))],
        [sg.Text("Select output folder:")],
        [sg.Input(), sg.FolderBrowse(key="OUTPUT_FOLDER")],
        [sg.Text("Enter output file name (without extension):")],
        [sg.Input(key="OUTPUT_FILE")],
        [sg.Text("Select output file type:")],
        [sg.Combo(["txt", "docx"], default_value="docx", key="FILE_TYPE")],
        [sg.Text("Select Whisper model version:")],
        [sg.Combo(["tiny", "base", "small", "medium", "large", "turbo"], default_value="tiny", key="MODEL")],
        [sg.Text("Select language that is spoken in the audio file:")],
        [sg.Combo(["nl", "en"], default_value="nl", key="LANGUAGE")],
        [sg.Button("Run Whisper"), sg.Button("Exit")],
        [sg.Text("", key="OUTPUT", size=(40, 3))],
    ]
    return layout

def validate_paths(audio_file, output_folder):
    if not os.path.isfile(audio_file):
        sg.popup("Error", f"Audio file does not exist: {audio_file}")
        return False
    if not os.path.isdir(output_folder):
        sg.popup("Error", f"Output folder does not exist: {output_folder}")
        return False
    return True

def load_whisper_model(model_version):
    try:
        model = whisper.load_model(model_version)
        return model
    except Exception as e:
        sg.popup("Error", f"Failed to load Whisper model: {str(e)}")
        return None

def transcribe_audio(model, audio_file, language):
    try:
        model_output = model.transcribe(audio_file, language=language)
        return model_output
    except Exception as e:
        sg.popup("Error", f"Transcription failed: {str(e)}")
        return None

def perform_diarization(audio_file, huggingface_token):
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=huggingface_token 
        )
        diarization = pipeline(audio_file)
        return diarization
    except Exception as e:
        sg.popup("Error", f"Speaker diarization failed: {str(e)}")
        return None

def add_speakers_to_transcription(segments, diarization):
    text = ""
    for segment in segments:
        start_time = segment["start"]
        segment_text = segment["text"]

        speaker = None
        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            if turn.start <= start_time <= turn.end:
                speaker = speaker_id
                break

        if speaker:
            text += f"[{speaker}]: {segment_text}\n"
        else:
            text += f"[Unknown]: {segment_text}\n"
    return text

def save_output_to_file(output_config, text):
    if output_config["file_type"] == "txt":
        try:
            output_file = f"{output_config['file_name']}.txt"
            output_path = os.path.join(output_config["folder"], output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            return True
        except Exception as e:
            sg.popup("Error", f"Saving txt file failed: {str(e)}")
            return False
    elif output_config["file_type"] == "docx":
        try:
            output_file = f"{output_config['file_name']}.docx"
            output_path = os.path.join(output_config["folder"], output_file)
            document = Document()
            document.add_paragraph(text)
            document.save(output_path)
            return True
        except Exception as e:
            sg.popup("Error", f"Saving docx file failed: {str(e)}")
            return False

def run_app(huggingface_token, window):
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if event == "Run Whisper":
            audio_file = values["AUDIO_FILE"]
            output_folder = values["OUTPUT_FOLDER"]
            output_file_name = values["OUTPUT_FILE"]
            output_file_type = values["FILE_TYPE"]
            model_version = values["MODEL"]
            language = values["LANGUAGE"]

            if not validate_paths(audio_file, output_folder):
                continue

            output_config = {
            "folder" : output_folder,
            "file_name" : output_file_name,
            "file_type" : output_file_type,
            }

            if not audio_file or not output_folder or not output_file_name:
                sg.popup("Error", "Please fill in all fields.")
                continue

            window["OUTPUT"].update("Loading Whisper model...")
            model = load_whisper_model(model_version)
            if model is None:
                continue

            window["OUTPUT"].update("Transcribing audio...")
            model_output = transcribe_audio(model, audio_file, language)
            if model_output is None:
                continue
            model_output_segments = model_output["segments"]

            window["OUTPUT"].update("Performing speaker diarization...")
            diarization = perform_diarization(audio_file, huggingface_token)
            if diarization is None:
                continue

            window["OUTPUT"].update("Adding speakers to transcription...")
            text = add_speakers_to_transcription(model_output_segments, diarization)

            window["OUTPUT"].update("Saving output file...")
            if not save_output_to_file(output_config, text):
                continue

            output_message = (
                f"Audio file: {audio_file}\n"
                f"Output folder: {output_folder}\n"
                f"Output file: {output_file_name}\n"
                f"Model version: {model_version}\n"
                f"Language: {language}\n"
                f"Transcription saved successfully!"
            )
            window["OUTPUT"].update(output_message)

    window.close()


def main():
    huggingface_token = load_huggingface_token("config.ini")
    layout = create_layout()
    window = sg.Window("Whisper App", layout)
    run_app(huggingface_token, window)

if __name__ == "__main__":
    main()