import whisper
from pyannote.audio import Pipeline


def load_whisper_model(model_version, error_popup_callback):
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


def transcribe_audio(model, audio_file, language, error_popup_callback):
    try:
        model_output = model.transcribe(audio_file, language=language)
        return model_output
    except Exception as e:
        error_popup_callback(f"Transcription failed: {str(e)}")
        return None


def perform_diarization(audio_file, hf_token, error_popup_callback):
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
        diarization = pipeline(audio_file)
        return diarization
    except Exception as e:
        error_popup_callback(f"Speaker diarization failed: {str(e)}")
        return None


def add_speakers_to_transcription(segments, diarization):
    text = ""
    for segment in segments:
        start_time = segment["start"]
        segment_text = segment["text"]

        speaker = None
        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            if turn.start <= start_time <= turn.end:
                speaker = speaker_id
                break

        if speaker:
            text += f"[{speaker}]: {segment_text}\n"
        else:
            text += f"[Unknown]: {segment_text}\n"
    return text


def run_transcription_pipeline(
    audio_processing_config, progress_callback, error_popup_callback
):
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
