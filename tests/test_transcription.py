from unittest.mock import Mock, patch

import whisper

from src.transcription import load_whisper_model, transcribe_audio


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


def test_transcribe_audio():
    test_model = Mock(spec=whisper.model.Whisper)
    test_audio_file = "filename.wav"
    test_language = "en"
    mock_callback = Mock()

    test_model.transcribe.return_value = "correct"

    assert (
        transcribe_audio(test_model, test_audio_file, test_language, mock_callback)
        == "correct"
    )
    mock_callback.assert_not_called()
