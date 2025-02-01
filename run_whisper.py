import os

import PySimpleGUI as sg
import whisper


model = whisper.load_model("tiny")

layout = [
    [sg.Text("Select an audio file:")],
    [sg.Input(), sg.FileBrowse(key="AUDIO_FILE", file_types=(("Audio Files", "*.wav *.mp3 *.flac"),))],
    [sg.Text("Select output folder:")],
    [sg.Input(), sg.FolderBrowse(key="OUTPUT_FOLDER")],
    [sg.Text("Enter output file name (without extension):")],
    [sg.Input(key="OUTPUT_FILE")],
    [sg.Text("Select Whisper model version:")],
    [sg.Combo(["tiny", "base", "small", "medium", "large", "turbo"], default_value="tiny", key="MODEL")],
    [sg.Combo(["nl", "en"], default_value="nl", key="LANGUAGE")],
    [sg.Button("Run Whisper"), sg.Button("Exit")],
    [sg.Text("", key="OUTPUT", size=(40, 3))],
]

window = sg.Window("Whisper App", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break

    if event == "Run Whisper":
        audio_file = values["AUDIO_FILE"]
        output_folder = values["OUTPUT_FOLDER"]
        output_name = values["OUTPUT_FILE"]
        model_version = values["MODEL"]
        language = values["LANGUAGE"]

        print(language)
        print(type(language))

        if not audio_file or not output_folder or not output_name:
            sg.popup("Error", "Please fill in all fields.")
            continue

        try:
            model_output = model.transcribe(audio_file, language=language)
            text = model_output["text"]
        except Exception as e:
            sg.popup("Error", f"Transcription failed: {str(e)}")
            continue

        output_file = f"{output_name}.txt"
        output_path = os.path.join(output_folder, output_file)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            sg.popup("Error", f"Failed to write output file: {str(e)}")
            continue

        output_message = (
            f"Audio file: {audio_file}\n"
            f"Output folder: {output_folder}\n"
            f"Output file: {output_file}\n"
            f"Model version: {model_version}\n"
            f"Language: {language}\n"
            f"Transcription saved successfully!"
        )
        window["OUTPUT"].update(output_message)

window.close()
