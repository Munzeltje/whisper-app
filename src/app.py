import PySimpleGUI as sg

from src.transcription import run_transcription_pipeline
from src.util import validate_user_input, save_output_to_file


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


def get_user_input(values, error_popup_callback):
    if not validate_user_input(
        values["AUDIO_FILE"],
        values["OUTPUT_FOLDER"],
        values["OUTPUT_FILE"],
        error_popup_callback,
    ):
        return None

    user_input = {
        "audio_file": values["AUDIO_FILE"],
        "output_folder": values["OUTPUT_FOLDER"],
        "output_file_name": values["OUTPUT_FILE"],
        "file_type": values["FILE_TYPE"],
        "model_version": values["MODEL"],
        "language": values["LANGUAGE"],
    }

    return user_input


def build_configs(user_input, hf_token):
    audio_processing_config = {
        "audio_file": user_input["audio_file"],
        "model_version": user_input["model_version"],
        "language": user_input["language"],
        "hf_token": hf_token,
    }

    output_config = {
        "folder": user_input["output_folder"],
        "file_name": user_input["output_file_name"],
        "file_type": user_input["file_type"],
    }

    return audio_processing_config, output_config


def save_transcription(output_config, text, update_ui, error_popup_callback):
    update_ui("Saving output file...")
    if not save_output_to_file(output_config, text, error_popup_callback):
        update_ui("Error: Saving output failed")
        return False
    update_ui("Transcription saved successfully!")
    return True


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
            user_input = get_user_input(values, popup)
            if user_input is None:
                continue

            audio_processing_config, output_config = build_configs(user_input, hf_token)

            text = run_transcription_pipeline(
                audio_processing_config,
                progress_callback=update_ui,
                error_popup_callback=popup,
            )
            if text is None:
                continue

            if not save_transcription(output_config, text, update_ui, popup):
                continue

    window.close()
