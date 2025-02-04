import tempfile
import docx
import pytest
from unittest.mock import Mock, patch, mock_open

from src.util import (
    load_huggingface_token,
    validate_user_input_types,
    validate_paths,
    validate_user_input,
    save_as_txt,
    save_as_docx,
    save_output_to_file,
)


def test_load_huggingface_token():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini") as temp_file:
        temp_file.write("[huggingface]\ntoken = test_token\n")
        temp_file.flush()

        test_token = load_huggingface_token(temp_file.name)
        expected_token = "test_token"

        assert test_token == expected_token


def test_load_huggingface_token_no_file():
    with pytest.raises(KeyError):
        load_huggingface_token("non_existent_file.ini")


def test_load_huggingface_token_no_token_in_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini") as temp_file:
        temp_file.write("[huggingface]\n")
        temp_file.flush()

        with pytest.raises(KeyError):
            load_huggingface_token(temp_file.name)


def test_validate_user_input_types():
    audio_file = "audio_file"
    output_folder = "output_folder"
    output_file_name = "output_file_name"

    assert validate_user_input_types(audio_file, output_folder, output_file_name)


def test_validate_user_input_is_complete_no_audio_file():
    audio_file = None
    output_folder = "output_folder"
    output_file_name = "output_file_name"

    assert not validate_user_input_types(audio_file, output_folder, output_file_name)


def test_validate_user_input_types_no_output_folder():
    audio_file = "audio_file"
    output_folder = None
    output_file_name = "output_file_name"

    assert not validate_user_input_types(audio_file, output_folder, output_file_name)


def test_validate_user_input_types_no_output_file_name():
    audio_file = "audio_file"
    output_folder = "output_folder"
    output_file_name = None

    assert not validate_user_input_types(audio_file, output_folder, output_file_name)


def test_validate_user_input_types_wrong_type():
    audio_file = "audio_file"
    output_folder = 10
    output_file_name = "output_file_name"

    assert not validate_user_input_types(audio_file, output_folder, output_file_name)


def test_validate_paths():
    temp_output_folder = tempfile.TemporaryDirectory()
    temp_output_folder_path = temp_output_folder.name

    mock_callback = Mock()

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name

        assert validate_paths(
            temp_audio_file_path, temp_output_folder_path, mock_callback
        )
        mock_callback.assert_not_called()

    temp_output_folder.cleanup()


def test_validate_paths_audio_file_does_not_exist():
    temp_audio_file = "this_file_does_not_exist.wav"

    temp_output_folder = tempfile.TemporaryDirectory()
    temp_output_folder_path = temp_output_folder.name

    mock_callback = Mock()

    assert not validate_paths(temp_audio_file, temp_output_folder_path, mock_callback)
    mock_callback.assert_called_once()


def test_validate_paths_audio_file_is_not_audio():
    temp_output_folder = tempfile.TemporaryDirectory()
    temp_output_folder_path = temp_output_folder.name

    mock_callback = Mock()

    with tempfile.NamedTemporaryFile(suffix=".png") as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name

        assert not validate_paths(
            temp_audio_file_path, temp_output_folder_path, mock_callback
        )
        mock_callback.assert_called_once()


def test_validate_paths_output_folder_is_not_a_folder():
    temp_output_folder_path = "this_folder/does_not_exist"

    mock_callback = Mock()

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name

        assert not validate_paths(
            temp_audio_file_path, temp_output_folder_path, mock_callback
        )
        mock_callback.assert_called_once()


@patch("src.util.validate_user_input_types")
@patch("src.util.validate_paths")
def test_validate_user_input(mock_validate_paths, mock_validate_types):
    mock_callback = Mock()

    mock_validate_types.return_value = True
    mock_validate_paths.return_value = True
    assert validate_user_input("audio.wav", "/output", "ouput_file.txt", mock_callback)
    mock_callback.assert_not_called()


@patch("src.util.validate_user_input_types")
@patch("src.util.validate_paths")
def test_validate_user_input_wrong_types(mock_validate_paths, mock_validate_types):
    mock_callback = Mock()

    mock_validate_types.return_value = False
    mock_validate_paths.return_value = True
    assert not validate_user_input(
        "audio.wav", "/output", "ouput_file.txt", mock_callback
    )
    mock_callback.assert_called_once()


@patch("src.util.validate_user_input_types")
@patch("src.util.validate_paths")
def test_validate_user_input_invalid_paths(mock_validate_paths, mock_validate_types):
    mock_callback = Mock()

    mock_validate_types.return_value = True
    mock_validate_paths.return_value = False

    assert not validate_user_input(
        "audio.wav", "/output", "ouput_file.txt", mock_callback
    )
    mock_callback.assert_called_once()


@patch("builtins.open", new_callable=mock_open)
def test_save_as_txt(mock_file):
    test_ouput_config = {
        "folder": "/folder",
        "file_name": "filename",
        "file_type": "txt",
    }
    test_text = "This would be a transcription."
    mock_callback = Mock()

    expected_save_path = "/folder/filename.txt"

    assert save_as_txt(test_ouput_config, test_text, mock_callback)
    mock_file.assert_called_once_with(expected_save_path, "w", encoding="utf-8")
    mock_callback.assert_not_called()


@patch("builtins.open", new_callable=mock_open)
def test_save_as_txt_permission_denied(mock_file_open):
    mock_file_open.side_effect = OSError("Permission denied")

    test_ouput_config = {
        "folder": "/folder",
        "file_name": "filename",
        "file_type": "txt",
    }
    test_text = "This would be a transcription."
    mock_callback = Mock()

    expected_save_path = "/folder/filename.txt"
    expected_message = "Saving txt file failed: Permission denied"

    assert not save_as_txt(test_ouput_config, test_text, mock_callback)
    mock_file_open.assert_called_once_with(expected_save_path, "w", encoding="utf-8")
    mock_callback.assert_called_once_with(expected_message)


@patch("src.util.Document")
def test_save_as_docx(mock_document_class):
    mock_document_instance = mock_document_class.return_value

    test_ouput_config = {
        "folder": "/folder",
        "file_name": "filename",
        "file_type": "docx",
    }
    test_text = "This would be a transcription."
    mock_callback = Mock()

    expected_save_path = "/folder/filename.docx"

    assert save_as_docx(test_ouput_config, test_text, mock_callback)
    mock_document_class.assert_called_once()
    mock_document_instance.add_paragraph.assert_called_once_with(test_text)
    mock_document_instance.save.assert_called_once_with(expected_save_path)
    mock_callback.assert_not_called()


@patch("src.util.Document")
def test_save_as_docx_permission_denied(mock_document_class):
    mock_document_instance = mock_document_class.return_value
    mock_document_instance.save.side_effect = OSError("Permission denied")

    test_ouput_config = {
        "folder": "/folder",
        "file_name": "filename",
        "file_type": "docx",
    }
    test_text = "This would be a transcription."
    mock_callback = Mock()

    expected_save_path = "/folder/filename.docx"
    expected_message = "Saving docx file failed: Permission denied"

    assert not save_as_docx(test_ouput_config, test_text, mock_callback)
    mock_document_class.assert_called_once()
    mock_document_instance.add_paragraph.assert_called_once_with(test_text)
    mock_document_instance.save.assert_called_once_with(expected_save_path)
    mock_callback.assert_called_once_with(expected_message)


@patch("src.util.save_as_docx")
@patch("src.util.save_as_txt")
def test_save_output_to_file_txt(mock_save_as_txt, mock_save_as_docx):
    test_ouput_config = {
        "folder": "/folder",
        "file_name": "filename",
        "file_type": "txt",
    }
    test_text = "This would be a transcription."
    mock_callback = Mock()

    mock_save_as_txt.return_value = True

    assert save_output_to_file(test_ouput_config, test_text, mock_callback)
    mock_save_as_txt.assert_called_once_with(
        test_ouput_config, test_text, mock_callback
    )
    mock_callback.assert_not_called()
    mock_save_as_docx.assert_not_called()


@patch("src.util.save_as_docx")
@patch("src.util.save_as_txt")
def test_save_output_to_file_docx(mock_save_as_txt, mock_save_as_docx):
    test_ouput_config = {
        "folder": "/folder",
        "file_name": "filename",
        "file_type": "docx",
    }
    test_text = "This would be a transcription."
    mock_callback = Mock()

    mock_save_as_docx.return_value = True

    assert save_output_to_file(test_ouput_config, test_text, mock_callback)
    mock_save_as_docx.assert_called_once_with(
        test_ouput_config, test_text, mock_callback
    )
    mock_callback.assert_not_called()
    mock_save_as_txt.assert_not_called()


def test_save_output_to_file_unsupported_file_type():
    test_ouput_config = {
        "folder": "/folder",
        "file_name": "filename",
        "file_type": "png",
    }
    test_text = "This would be a transcription."
    mock_callback = Mock()

    assert not save_output_to_file(test_ouput_config, test_text, mock_callback)
    mock_callback.assert_called_once_with("Filetype is not supported: png")
