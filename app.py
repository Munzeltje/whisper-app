import PySimpleGUI as sg

from transcription import run_transcription_pipeline
from util import validate_user_input, save_output_to_file


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
