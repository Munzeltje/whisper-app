from typing import Callable

import PySimpleGUI as sg

from src.transcription import run_transcription_pipeline
from src.util import validate_user_input, save_output_to_file


def create_layout() -> list[list[sg.Element]]:
    """
    Creates the UI for the app.

    Returns:
        list[list[sg.Element]]: a list of PySimpleGUI elements
    """
    layout = [
        [sg.Text("Select an audio file:")],
        [
            sg.Input(),
            sg.FileBrowse(
                key="AUDIO_FILE",
                file_types=(
                    (
                        "Audio Files",
                        "*.wav *.mp3 *.flac *.ogg *.mp4 *.m4a *.aiff *.caf",
                    ),
                ),
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


def get_user_input(values: dict, error_popup_callback: Callable) -> dict | None:
    """
    Takes values obtained through window.read and parse the user input. Calls callback
    to throw an error and returns None in case of invalid input.

    Args:
        values (dict): values obtained through UI
        error_popup_callback (Callable[str]): show user a popup with an error message
            if input is invalid

    Returns:
        dict | None: either a dict with the parsed input if successful;
            otherwise, None if an exception occurs.
    """
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


def build_configs(user_input: dict, hf_token: str) -> tuple[dict, dict]:
    """
    Creates dicts to neatly keep track of information regarding the audio processing and
    the saving of the output.

    Args:
        user_input (dict): input obtained from user, filepaths
        hf_token (str): Hugging Face API token.

    Returns:
        tuple[dict, dict]: created configs
    """
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


def save_transcription(
    output_config: dict, text: str, update_ui: Callable, error_popup_callback: Callable
) -> bool:
    """
    Calls save function and update the UI with progress.

    Args:
        output_config (dict): contains file information
        text (str): the text that will be saved to the file
        update_ui (Callable[str]): callback used to update UI with progress
        error_popup_callback (Callable[str]): callback that creates popup in case of an error

    Returns:
        bool: whether or not transcription was saved successfully
    """
    update_ui("Saving output file...")
    if not save_output_to_file(output_config, text, error_popup_callback):
        update_ui("Error: Saving output failed")
        return False
    update_ui("Transcription saved successfully!")
    return True


def run_app(hf_token: str, window: sg.Window) -> None:
    """
    Reads user interaction with UI and respond accordingly, putting the flow
    of the app together. Keep user informed of progress by updating UI throughout
    the process.

    Args:
        hf_token (str): Hugging Face API token.
        window (sg.Window): UI of the app
    """

    def update_ui(message: str) -> None:
        window["OUTPUT"].update(message)

    def popup(message: str) -> None:
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

            saved = save_transcription(output_config, text, update_ui, popup)
            if not saved:
                continue

    window.close()
