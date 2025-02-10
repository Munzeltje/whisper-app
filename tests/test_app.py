from unittest.mock import Mock, MagicMock, patch

import PySimpleGUI as sg

from src.app import (
    create_layout,
    get_user_input,
    build_configs,
    save_transcription,
    run_app,
)


def test_create_layout_returns_list_of_gui_elements():
    layout = create_layout()

    assert isinstance(layout, list)
    assert all(isinstance(row, list) for row in layout)
    assert all(isinstance(el, sg.Element) for row in layout for el in row)


def test_create_layout_valid_elements_and_keys():
    layout = create_layout()

    # Flatten the nested list structure to search easily
    elements = [el for row in layout for el in row]

    expected_elements = [
        "AUDIO_FILE",
        "OUTPUT_FOLDER",
        "OUTPUT_FILE",
        "FILE_TYPE",
        "MODEL",
        "LANGUAGE",
        "OUTPUT",
    ]

    expected_buttons = ["Run Whisper", "Exit"]

    for expected_element in expected_elements:
        assert any(expected_element == element.key for element in elements), f"Missing element: {expected_element}"

    assert any(
        isinstance(element, sg.Button) and element.ButtonText in expected_buttons
        for element in elements
    ), "Missing buttons"

    keys = [el.key for el in elements if hasattr(el, "key") and el.key is not None]
    assert len(keys) == len(set(keys)), "Duplicate keys found in layout"
    assert all(isinstance(key, str) for key in keys), "All keys should be strings"


@patch("src.app.validate_user_input", return_value=True)
def test_get_user_input(mock_validate):
    test_values = {
        "AUDIO_FILE" : "file.wav",
        "OUTPUT_FOLDER" : "output/folder",
        "OUTPUT_FILE" : "output.txt",
        "FILE_TYPE" : "txt",
        "MODEL" : "tiny",
        "LANGUAGE" : "en",
    }

    mock_callback = Mock()

    expected = {
        "audio_file" : "file.wav",
        "output_folder" : "output/folder",
        "output_file_name" : "output.txt",
        "file_type" : "txt",
        "model_version" : "tiny",
        "language" : "en",
    }

    assert get_user_input(test_values, mock_callback) == expected
    mock_validate.assert_called_once()
    mock_callback.assert_not_called()


@patch("src.app.validate_user_input", return_value=False)
def test_get_user_input_invalid_input(mock_validate):
    test_values = {
        "AUDIO_FILE" : "file.wav",
        "OUTPUT_FOLDER" : "output/folder",
        "OUTPUT_FILE" : "output.txt",
        "FILE_TYPE" : "txt",
        "MODEL" : "tiny",
        "LANGUAGE" : "en",
    }

    mock_callback = Mock()

    assert get_user_input(test_values, mock_callback) is None
    mock_validate.assert_called_once()


def test_build_configs():
    test_user_input = {
        "audio_file": "file.wav",
        "output_folder": "output/folder",
        "output_file_name": "output",
        "file_type": "txt",
        "model_version": "tiny",
        "language": "en",
    }
    
    test_hf_token = "mock_token"

    expected_audio_config = {
        "audio_file": "file.wav",
        "model_version": "tiny",
        "language": "en",
        "hf_token": test_hf_token,
    }

    expected_output_config = {
        "folder": "output/folder",
        "file_name": "output",
        "file_type": "txt",
    }

    audio_config, output_config = build_configs(test_user_input, test_hf_token)

    assert audio_config == expected_audio_config, "Audio config does not match expected"
    assert output_config == expected_output_config, "Output config does not match expected"


@patch("src.app.save_output_to_file", return_value=True)
def test_save_transcription_success(mock_save_output):
    mock_update_ui = Mock()
    mock_popup = Mock()

    output_config = {
        "folder": "output/folder",
        "file_name": "output",
        "file_type": "txt",
    }
    text = "Transcribed text"

    assert save_transcription(output_config, text, mock_update_ui, mock_popup)

    mock_update_ui.assert_any_call("Saving output file...")
    mock_update_ui.assert_any_call("Transcription saved successfully!")

    mock_save_output.assert_called_once_with(output_config, text, mock_popup)


@patch("src.app.save_output_to_file", return_value=False)
def test_save_transcription_failure(mock_save_output):
    mock_update_ui = Mock()
    mock_popup = Mock()

    output_config = {
        "folder": "output/folder",
        "file_name": "output",
        "file_type": "txt",
    }
    text = "Transcribed text"

    assert not save_transcription(output_config, text, mock_update_ui, mock_popup)

    mock_update_ui.assert_any_call("Saving output file...")
    mock_update_ui.assert_any_call("Error: Saving output failed")

    mock_save_output.assert_called_once_with(output_config, text, mock_popup)


@patch("src.app.get_user_input")
@patch("src.app.build_configs")
@patch("src.app.run_transcription_pipeline")
@patch("src.app.save_transcription")
def test_run_app(
    mock_save_transcription, mock_run_pipeline, mock_build_configs, mock_get_user_input
):
    mock_window = Mock()
    mock_window.read.side_effect = [("Run Whisper", {}), (sg.WIN_CLOSED, {})]

    mock_get_user_input.return_value = {
        "audio_file": "test.wav",
        "output_folder": "/output",
        "output_file_name": "transcription",
        "file_type": "txt",
        "model_version": "base",
        "language": "en",
    }

    mock_build_configs.return_value = (
        {"audio_file": "test.wav", "model_version": "base", "language": "en", "hf_token": "mock_hf_token"},
        {"folder": "/output", "file_name": "transcription", "file_type": "txt"},
    )

    run_app("mock_hf_token", mock_window)

    mock_get_user_input.assert_called_once()
    mock_build_configs.assert_called_once()
    mock_run_pipeline.assert_called_once()
    mock_save_transcription.assert_called_once()


def test_run_app_exits_on_close():
    mock_window = Mock()
    mock_window.read.side_effect = [(sg.WIN_CLOSED, {})]

    run_app("mock_hf_token", mock_window)

    mock_window.read.assert_called()
    mock_window.close.assert_called_once()


@patch("src.app.get_user_input", return_value=None)
@patch("src.app.build_configs")
@patch("src.app.run_transcription_pipeline")
@patch("src.app.save_transcription")
def test_run_app_handles_input_validation_failure(
    mock_save_transcription, mock_run_pipeline, mock_build_configs, mock_get_user_input
):
    mock_window = Mock()

    mock_window.read.side_effect = [
        ("Run Whisper", {}),
        (sg.WIN_CLOSED, {}),
    ]

    run_app("mock_hf_token", mock_window)

    mock_get_user_input.assert_called_once()

    mock_build_configs.assert_not_called()
    mock_run_pipeline.assert_not_called()
    mock_save_transcription.assert_not_called()


@patch("src.app.get_user_input")
@patch("src.app.build_configs")
@patch("src.app.run_transcription_pipeline", return_value=None)
@patch("src.app.save_transcription")
def test_run_app_handles_transcription_failure(
    mock_save_transcription, mock_run_pipeline, mock_build_configs, mock_get_user_input
):
    mock_window = Mock()
    mock_window.read.side_effect = [("Run Whisper", {}), (sg.WIN_CLOSED, {})]

    mock_get_user_input.return_value = {
        "audio_file": "test.wav",
        "output_folder": "/output",
        "output_file_name": "transcription",
        "file_type": "txt",
        "model_version": "base",
        "language": "en",
    }

    mock_build_configs.return_value = (
        {"audio_file": "test.wav", "model_version": "base", "language": "en", "hf_token": "mock_hf_token"},
        {"folder": "/output", "file_name": "transcription", "file_type": "txt"},
    )

    run_app("mock_hf_token", mock_window)

    mock_get_user_input.assert_called_once()
    mock_build_configs.assert_called_once()
    mock_run_pipeline.assert_called_once()
    mock_save_transcription.assert_not_called()


@patch("src.app.get_user_input")
@patch("src.app.build_configs")
@patch("src.app.run_transcription_pipeline")
@patch("src.app.save_transcription", return_value=False)
def test_run_app_save_failed(
    mock_save_transcription, mock_run_pipeline, mock_build_configs, mock_get_user_input
):
    mock_window = Mock()
    mock_window.read.side_effect = [("Run Whisper", {}), (sg.WIN_CLOSED, {})]

    mock_get_user_input.return_value = {
        "audio_file": "test.wav",
        "output_folder": "/output",
        "output_file_name": "transcription",
        "file_type": "txt",
        "model_version": "base",
        "language": "en",
    }

    mock_build_configs.return_value = (
        {"audio_file": "test.wav", "model_version": "base", "language": "en", "hf_token": "mock_hf_token"},
        {"folder": "/output", "file_name": "transcription", "file_type": "txt"},
    )

    run_app("mock_hf_token", mock_window)

    mock_get_user_input.assert_called_once()
    mock_build_configs.assert_called_once()
    mock_run_pipeline.assert_called_once()
    mock_save_transcription.assert_called_once()
