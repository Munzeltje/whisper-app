from unittest.mock import Mock, patch

import PySimpleGUI as sg

from src.app import (
    create_layout,
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


def test_run_app_exits_on_close():
    mock_window = Mock()
    mock_window.read.side_effect = [(sg.WIN_CLOSED, {})]

    run_app("mock_hf_token", mock_window)

    mock_window.read.assert_called()
    mock_window.close.assert_called_once()