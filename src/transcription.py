from typing import Callable

import whisper
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation


def load_whisper_model(
    model_version: str, error_popup_callback: Callable
) -> whisper.Whisper | None:
    """
    Loads correct version of Whisper from OpenAI

    Args:
        model_version (str): size of the model that will be used
        error_popup_callback (Callable[str]): callback that creates popup in case of an error

    Returns:
        whisper.Whisper | None: Whisper model if it is loaded successfully; otherwise,
            None if an exception occurs.
    """
    if not model_version in ("tiny", "base", "small", "medium", "large", "turbo"):
        error_popup_callback(
            f"Failed to load Whisper model: Invalid model version: {model_version}"
        )
        return None
    try:
        model = whisper.load_model(model_version)
        return model
    except Exception as e:
        error_popup_callback(f"Failed to load Whisper model: {str(e)}")
        return None


def transcribe_audio(
    model: whisper.Whisper,
    audio_file: str,
    language: str,
    error_popup_callback: Callable,
) -> dict[str, str | list] | None:
    """
    Transcribes the given audio file using a preloaded Whisper model and returns the
    transcription output.

    Args:
        model (whisper.Whisper): A Whisper model instance used for transcription.
        audio_file (str): Path to the audio file to be transcribed. Supported formats
            include mp3, wav, ogg, flac, mp4, m4a, aiff, and caf.
        language (str): Language code (e.g., "en", "nl") specifying the spoken language.
        error_popup_callback (Callable[str]): A callback function used to display error
            messages to the user as a popup.

    Returns:
        dict[str, str | list] | None: A dictionary containing the transcription result
            if successful; otherwise, None if an exception occurs.
    """
    try:
        model_output = model.transcribe(audio_file, language=language)
        return model_output
    except Exception as e:
        error_popup_callback(f"Transcription failed: {str(e)}")
        return None


def perform_diarization(
    audio_file: str,
    hf_token: str,
    error_popup_callback: Callable,
    model_name: str = "pyannote/speaker-diarization-3.1",
) -> Annotation | None:
    """
    Performs speaker diarization on an audio file using a pretrained PyAnnote pipeline.

    Args:
        audio_file (str): Path to the audio file to process.
        hf_token (str): Hugging Face API token for access to the diarization model.
        error_popup_callback (Callable[str]): Callback function to show an error message when diarization fails.
        model_name (str, optional): Name of the pretrained PyAnnote diarization model to use.
            Defaults to "pyannote/speaker-diarization-3.1".

    Returns:
        Annotation | None: A PyAnnote Annotation object containing speaker turn information if successful;
            otherwise, None if an exception occurs.
    """
    try:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        diarization = pipeline(audio_file)
        return diarization
    except Exception as e:
        error_popup_callback(f"Speaker diarization failed: {str(e)}")
        return None


def add_speakers_to_transcription(segments: list[dict], diarization: Annotation) -> str:
    """
    Assigns speaker labels to transcribed segments based on diarization output and
    formats them as readable text.

    Args:
        segments (list[dict]): A list of transcribed segments from Whisper, where each
            segment is a dictionary.
        diarization (Annotation): A PyAnnote Annotation object containing speaker turn
            information with time spans and associated speaker labels.

    Returns:
        str: A formatted string where each line is prefixed with a speaker label (e.g., "[Speaker 1]: Hello world").
            Segments that do not fall within any diarization turn are labeled as "[Unknown]".
    """
    speaker_map = {
        (turn.start, turn.end): speaker_id
        for turn, _, speaker_id in diarization.itertracks(yield_label=True)
    }

    text = []
    for segment in segments:
        speaker = next(
            (
                speaker_id
                for (start, end), speaker_id in speaker_map.items()
                if start <= segment["start"] <= end
            ),
            "Unknown",
        )
        text.append(f"[{speaker}]: {segment['text']}")
    text = "\n".join(text)
    return text


def run_transcription_pipeline(
    audio_processing_config: dict,
    progress_callback: Callable,
    error_popup_callback: Callable,
) -> str | None:
    """
    Executes the full audio transcription pipeline, including Whisper transcription and PyAnnote speaker diarization.

    Args:
        audio_processing_config (dict): A dictionary containing configuration for audio processing. Expected keys:
            - "audio_file" (str): Path to the audio file to be transcribed.
            - "model_version" (str): Whisper model version to use (e.g., "tiny", "base").
            - "language" (str): Language code of the spoken language in the audio.
            - "hf_token" (str): Hugging Face token for accessing the diarization model.
        progress_callback (Callable[str]): A function to update the user interface or console with status messages.
        error_popup_callback (Callable[str]): A function to show error messages to the user, typically as a popup.

    Returns:
        str | None: A formatted transcription string with speaker labels if successful; otherwise, None if any step fails.
    """
    progress_callback("Loading Whisper model...")
    model = load_whisper_model(
        audio_processing_config["model_version"], error_popup_callback
    )
    if model is None:
        progress_callback("Error: Failed to load model")
        return None

    progress_callback("Transcribing audio...")
    model_output = transcribe_audio(
        model,
        audio_processing_config["audio_file"],
        audio_processing_config["language"],
        error_popup_callback,
    )
    if model_output is None:
        progress_callback("Error: Failed to transcribe audio.")
        return None

    progress_callback("Performing speaker diarization...")
    diarization = perform_diarization(
        audio_processing_config["audio_file"],
        audio_processing_config["hf_token"],
        error_popup_callback,
    )
    if diarization is None:
        progress_callback("Error: Diarization failed.")
        return None

    progress_callback("Adding speakers to transcription...")
    text = add_speakers_to_transcription(model_output["segments"], diarization)
    return text
