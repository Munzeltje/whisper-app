from unittest.mock import Mock, patch

import whisper
from pyannote.audio import Pipeline

from src.transcription import (
    load_whisper_model,
    transcribe_audio,
    perform_diarization,
    add_speakers_to_transcription,
)


@patch("whisper.load_model")
def test_load_whisper_model(mock_load_model):
    test_model_version = "tiny"
    mock_callback = Mock()

    result = load_whisper_model(test_model_version, mock_callback)
    expected = mock_load_model.return_value

    result == expected
    mock_callback.assert_not_called()


@patch("whisper.load_model")
def test_load_whisper_model_invalid_model_version(mock_load_model):
    test_model_version = "invalid_version"
    mock_callback = Mock()

    result = load_whisper_model(test_model_version, mock_callback)

    assert result is None
    mock_callback.assert_called_once_with(
        "Failed to load Whisper model: Invalid model version: invalid_version"
    )


@patch("whisper.load_model")
def test_load_whisper_model_exception(mock_load_model):
    test_model_version = "tiny"
    mock_callback = Mock()

    mock_load_model.side_effect = Exception("Something went wrong")

    result = load_whisper_model(test_model_version, mock_callback)

    assert result is None
    mock_callback.assert_called_once_with(
        "Failed to load Whisper model: Something went wrong"
    )


def test_transcribe_audio():
    test_model = Mock(spec=whisper.model.Whisper)
    test_audio_file = "filename.wav"
    test_language = "en"
    mock_callback = Mock()

    test_model.transcribe.return_value = "correct"

    result = transcribe_audio(test_model, test_audio_file, test_language, mock_callback)

    assert result == "correct"
    mock_callback.assert_not_called()


@patch.object(Pipeline, "from_pretrained")
def test_perform_diarization(mock_from_pretrained):
    test_audio_file = "filename.wav"
    test_hf_token = "token"
    mock_callback = Mock()

    mock_pipeline = Mock()
    mock_from_pretrained.return_value = mock_pipeline
    mock_pipeline.return_value = "diarization"

    result = perform_diarization(test_audio_file, test_hf_token, mock_callback)
    expected = "diarization"

    assert result == expected
    mock_callback.assert_not_called()


@patch.object(Pipeline, "from_pretrained")
def test_perform_diarization_exception(mock_from_pretrained):
    test_audio_file = "filename.wav"
    test_hf_token = "token"
    mock_callback = Mock()

    mock_from_pretrained.side_effect = Exception("exception")

    result = perform_diarization(test_audio_file, test_hf_token, mock_callback)

    assert result is None
    mock_callback.assert_called_once_with("Speaker diarization failed: exception")


def test_add_speakers_to_transcription():
    segments = [
        {"start": 0.5, "text": "Hello"},
        {"start": 2.0, "text": "How are you?"},
        {"start": 5.5, "text": "Goodbye"},
    ]

    mock_diarization = Mock()
    mock_diarization.itertracks.return_value = [
        (Mock(start=0.0, end=3.0), None, "Speaker 1"),
        (Mock(start=4.0, end=6.0), None, "Speaker 2"),
    ]

    result = add_speakers_to_transcription(segments, mock_diarization)

    expected = "[Speaker 1]: Hello\n[Speaker 1]: How are you?\n[Speaker 2]: Goodbye\n"

    assert result == expected


def test_add_speakers_to_transcription_no_match():
    segments = [
        {"start": 10.0, "text": "This is a test."},
    ]

    mock_diarization = Mock()
    mock_diarization.itertracks.return_value = [
        (Mock(start=0.0, end=5.0), None, "Speaker 1"),
    ]

    result = add_speakers_to_transcription(segments, mock_diarization)

    expected = "[Unknown]: This is a test.\n"

    assert result == expected


def test_add_speakers_to_transcription_empty_segments():
    mock_diarization = Mock()
    mock_diarization.itertracks.return_value = [
        (Mock(start=0.0, end=5.0), None, "Speaker 1"),
    ]

    result = add_speakers_to_transcription([], mock_diarization)

    assert result == ""


def test_add_speakers_to_transcription_empty_diarization():
    segments = [
        {"start": 1.0, "text": "Hello"},
    ]

    mock_diarization = Mock()
    mock_diarization.itertracks.return_value = []

    result = add_speakers_to_transcription(segments, mock_diarization)

    expected_output = "[Unknown]: Hello\n"

    assert result == expected_output
