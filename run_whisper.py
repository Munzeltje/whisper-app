import os
import configparser

from docx import Document
import PySimpleGUI as sg
import whisper
from pyannote.audio import Pipeline

from util import load_huggingface_token, validate_user_input, save_output_to_file


def create_layout():
    layout = [
        [sg.Text("Select an audio file:")],
        [
            sg.Input(),
            sg.FileBrowse(
                key="AUDIO_FILE", file_types=(("Audio Files", "*.wav *.mp3 *.flac"),)
            ),
        ],
        [sg.Text("Select output folder:")],
        [sg.Input(), sg.FolderBrowse(key="OUTPUT_FOLDER")],
        [sg.Text("Enter output file name (without extension):")],
        [sg.Input(key="OUTPUT_FILE")],
        [sg.Text("Select output file type:")],
        [sg.Combo(["txt", "docx"], default_value="docx", key="FILE_TYPE")],
        [sg.Text("Select Whisper model version:")],
        [
            sg.Combo(
                ["tiny", "base", "small", "medium", "large", "turbo"],
                default_value="tiny",
                key="MODEL",
            )
        ],
        [sg.Text("Select language that is spoken in the audio file:")],
        [sg.Combo(["nl", "en"], default_value="nl", key="LANGUAGE")],
        [sg.Button("Run Whisper"), sg.Button("Exit")],
        [sg.Text("", key="OUTPUT", size=(40, 3))],
    ]
    return layout


def load_whisper_model(model_version, error_popup_callback):
    try:
        model = whisper.load_model(model_version)
        return model
    except Exception as e:
        error_popup_callback(f"Failed to load Whisper model: {str(e)}")
        return None


def transcribe_audio(model, audio_file, language, error_popup_callback):
    try:
        model_output = model.transcribe(audio_file, language=language)
        return model_output
    except Exception as e:
        error_popup_callback(f"Transcription failed: {str(e)}")
        return None


def perform_diarization(audio_file, hf_token, error_popup_callback):
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
        diarization = pipeline(audio_file)
        return diarization
    except Exception as e:
        error_popup_callback(f"Speaker diarization failed: {str(e)}")
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


def run_transcription_pipeline(
    audio_processing_config, progress_callback, error_popup_callback
):
    progress_callback("Loading Whisper model...")
    model = load_whisper_model(
        audio_processing_config["model_version"], error_popup_callback
    )
    if model is None:
        progress_callback("Error: Failed to load model")
        return None

    progress_callback("Transcribing audio...")
    model_output = transcribe_audio(
        model,
        audio_processing_config["audio_file"],
        audio_processing_config["language"],
        error_popup_callback,
    )
    if model_output is None:
        progress_callback("Error: Failed to transcribe audio.")
        return None

    progress_callback("Performing speaker diarization...")
    diarization = perform_diarization(
        audio_processing_config["audio_file"],
        audio_processing_config["hf_token"],
        error_popup_callback,
    )
    if diarization is None:
        progress_callback("Error: Diarization failed.")
        return None

    progress_callback("Adding speakers to transcription...")
    text = add_speakers_to_transcription(model_output["segments"], diarization)

    return text


def run_app(hf_token, window):
    def update_ui(message):
        window["OUTPUT"].update(message)

    def popup(message):
        sg.popup("Error", message)

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

            if not validate_user_input(
                audio_file, output_folder, output_file_name, callback=popup
            ):
                continue

            audio_processing_config = {
                "audio_file": audio_file,
                "model_version": model_version,
                "language": language,
                "hf_token": hf_token,
            }

            output_config = {
                "folder": output_folder,
                "file_name": output_file_name,
                "file_type": output_file_type,
            }

            text = run_transcription_pipeline(
                audio_processing_config,
                progress_callback=update_ui,
                error_popup_callback=popup,
            )
            if text is None:
                continue

            update_ui("Saving output file...")
            if not save_output_to_file(output_config, text, error_popup_callback=popup):
                update_ui("Error: Saving output failed")
                continue
            update_ui("Transcription saved successfully!")
    window.close()


def main():
    hf_token = load_huggingface_token("config.ini")
    layout = create_layout()
    window = sg.Window("Whisper App", layout)
    run_app(hf_token, window)


if __name__ == "__main__":
    main()
