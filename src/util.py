import os
import configparser
from typing import Callable

from docx import Document


def load_huggingface_token(config_file: str) -> str:
    """
    Loads the Hugging Face API token from a configuration file.

    Args:
        config_file (str): Path to a .ini configuration file that must contain
            a [huggingface] section with a 'token' key.

    Returns:
        str : The Hugging Face API token.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If the required section or key is missing.
    """
    config = configparser.ConfigParser()

    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    config.read(config_file)

    if "huggingface" not in config or "token" not in config["huggingface"]:
        raise KeyError(
            f"'token' key not found in 'huggingface' section of {config_file}"
        )

    token = config["huggingface"]["token"]
    return token


def validate_user_input_types(
    audio_file: str, output_folder: str, output_file_name: str
) -> bool:
    """
    Validates that all provided user inputs are strings.

    Args:
        audio_file (str): The path to the audio file as provided by the user.
        output_folder (str): The path to the output directory as provided by the user.
        output_file_name (str): The desired name of the output file as provided by the user.

    Returns:
        bool: True if all inputs are strings; False otherwise.
    """
    for user_input in (audio_file, output_folder, output_file_name):
        if not isinstance(user_input, str):
            return False
    return True


def validate_paths(audio_file: str, output_folder: str, callback: Callable) -> bool:
    """
    Validates the existence and format of the audio file and the existence of the output folder.

    Args:
        audio_file (str): Path to the audio file to validate.
        output_folder (str): Path to the output folder to validate.
        callback (Callable[str]): Function to call with an error message if validation fails.

    Returns:
        bool: True if all paths are valid and the audio file format is supported; False otherwise.
    """
    if not os.path.isfile(audio_file):
        callback(f"Audio file does not exist: {audio_file}")
        return False

    audio_file_suffix = audio_file.split(".")[-1]
    if audio_file_suffix not in (
        "mp3",
        "wav",
        "ogg",
        "flac",
        "mp4",
        "m4a",
        "aiff",
        "caf",
    ):
        callback(f"File format is not supported: {audio_file_suffix}")
        return False

    if not os.path.isdir(output_folder):
        callback(f"Output folder does not exist: {output_folder}")
        return False

    return True


def validate_user_input(
    audio_file: str, output_folder: str, output_file_name: str, callback: Callable
):
    """
    Validates user input for audio transcription by checking types and file/folder existence.

    Args:
        audio_file (str): Path to the audio file to be transcribed.
        output_folder (str): Path to the folder where the output file should be saved.
        output_file_name (str): Desired name for the output file (without extension).
        callback (Callable[str]): Function to display an error message if validation fails.

    Returns:
        bool: True if all inputs are valid; False otherwise.
    """
    if not validate_user_input_types(audio_file, output_folder, output_file_name):
        callback("Please fill in all fields.")
        return False
    if not validate_paths(audio_file, output_folder, callback):
        callback("Please make sure all given paths are valid.")
        return False
    return True


def save_as_txt(output_config: dict, text: str, error_popup_callback: Callable):
    """
    Saves the given transcription text to a .txt file in the specified output location.

    Args:
        output_config (dict): A dictionary with output settings. Expected keys:
            - "folder": Path to the output folder.
            - "file_name": Name of the output file without extension.
        text (str): The transcription text to be written to the file.
        error_popup_callback (Callable[str]): Function to call with an error message if saving fails.

    Returns:
        bool: True if the file was saved successfully; False otherwise.
    """
    try:
        output_file = f"{output_config['file_name']}.txt"
        output_path = os.path.join(output_config["folder"], output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        error_popup_callback(f"Saving txt file failed: {str(e)}")
        return False


def save_as_docx(output_config: dict, text: str, error_popup_callback: Callable):
    """
    Saves the given transcription text to a .docx (Word) file in the specified output location.

    Args:
        output_config (dict): A dictionary with output settings. Expected keys:
            - "folder": Path to the output folder.
            - "file_name": Name of the output file without extension.
        text (str): The transcription text to be written to the .docx file.
        error_popup_callback (Callable[str]): Function to call with an error message if saving fails.

    Returns:
        bool: True if the file was saved successfully; False otherwise.
    """
    try:
        output_file = f"{output_config['file_name']}.docx"
        output_path = os.path.join(output_config["folder"], output_file)
        document = Document()
        document.add_paragraph(text)
        document.save(output_path)
        return True
    except Exception as e:
        error_popup_callback(f"Saving docx file failed: {str(e)}")
        return False


def save_output_to_file(output_config: dict, text: str, error_popup_callback: Callable):
    """
    Saves the transcription text to a file of the selected file type (.txt or .docx).

    Args:
        output_config (dict): A dictionary containing output settings. Expected keys:
            - "folder": Path to the output folder.
            - "file_name": Name of the output file without extension.
            - "file_type": File type to save ("txt" or "docx").
        text (str): The transcription text to save.
        error_popup_callback (Callable[str]): Function to call with an error message if saving fails or the file type is unsupported.

    Returns:
        bool: True if the file was saved successfully; False if saving failed or the file type is unsupported.
    """
    if output_config["file_type"] == "txt":
        saved = save_as_txt(output_config, text, error_popup_callback)
    elif output_config["file_type"] == "docx":
        saved = save_as_docx(output_config, text, error_popup_callback)
    else:
        error_popup_callback(f"Filetype is not supported: {output_config['file_type']}")
        saved = False
    return saved
