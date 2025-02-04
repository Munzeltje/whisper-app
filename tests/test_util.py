import tempfile
import pytest
from unittest.mock import Mock, patch

from src.util import (
    load_huggingface_token,
    validate_user_input_types,
    validate_paths,
    validate_user_input,
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
