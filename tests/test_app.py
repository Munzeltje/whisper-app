from unittest.mock import Mock, patch

import PySimpleGUI as sg

from src.app import (
    create_layout,
    get_user_input,
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


def test_run_app_exits_on_close():
    mock_window = Mock()
    mock_window.read.side_effect = [(sg.WIN_CLOSED, {})]

    run_app("mock_hf_token", mock_window)

    mock_window.read.assert_called()
    mock_window.close.assert_called_once()