import os
import configparser

from docx import Document


def load_huggingface_token(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    hf_token = config["huggingface"]["token"]
    return hf_token


def validate_user_input_is_complete(audio_file, output_folder, output_file_name):
    if not audio_file or not output_folder or not output_file_name:
        return False
    return True


def validate_paths(audio_file, output_folder, callback):
    if not os.path.isfile(audio_file):
        callback(f"Audio file does not exist: {audio_file}")
        return False
    if not os.path.isdir(output_folder):
        callback(f"Output folder does not exist: {output_folder}")
        return False
    return True


def validate_user_input(audio_file, output_folder, output_file_name, callback):
    if not validate_user_input_is_complete(audio_file, output_folder, output_file_name):
        callback("Please fill in all fields.")
        return False
    if not validate_paths(audio_file, output_folder, callback):
        callback("Please make sure all given paths are valid.")
        return False
    return True


def save_as_txt(output_config, text, error_popup_callback):
    try:
        output_file = f"{output_config['file_name']}.txt"
        output_path = os.path.join(output_config["folder"], output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        error_popup_callback(f"Saving txt file failed: {str(e)}")
        return False


def save_as_docx(output_config, text, error_popup_callback):
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


def save_output_to_file(output_config, text, error_popup_callback):
    if output_config["file_type"] == "txt":
        saved = save_as_txt(output_config, text, error_popup_callback)
    elif output_config["file_type"] == "docx":
        saved = save_as_docx(output_config, text, error_popup_callback)
    return saved
