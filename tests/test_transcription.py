from unittest.mock import Mock, patch

from src.transcription import load_whisper_model


@patch("whisper.load_model")
def test_load_whisper_model(mock_load_model):
    test_model_version = "tiny"
    mock_callback = Mock()

    assert (
        load_whisper_model(test_model_version, mock_callback)
        == mock_load_model.return_value
    )
    mock_callback.assert_not_called()


@patch("whisper.load_model")
def test_load_whisper_model_invalid_model_version(mock_load_model):
    test_model_version = "invalid_version"
    mock_callback = Mock()

    assert load_whisper_model(test_model_version, mock_callback) is None
    mock_callback.assert_called_once_with(
        "Failed to load Whisper model: Invalid model version: invalid_version"
    )


@patch("whisper.load_model")
def test_load_whisper_model_exception(mock_load_model):
    test_model_version = "tiny"
    mock_callback = Mock()

    mock_load_model.side_effect = Exception("Something went wrong")

    assert load_whisper_model(test_model_version, mock_callback) is None
    mock_callback.assert_called_once_with(
        "Failed to load Whisper model: Something went wrong"
    )
