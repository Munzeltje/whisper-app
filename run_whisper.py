import os

from docx import Document
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
    [sg.Text("Select output file type:")],
    [sg.Combo(["txt", "docx"], default_value="docx", key="FILE_TYPE")],
    [sg.Text("Select Whisper model version:")],
    [sg.Combo(["tiny", "base", "small", "medium", "large", "turbo"], default_value="tiny", key="MODEL")],
    [sg.Text("Select language that is spoken in the audio file:")],
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
        output_file_type = values["FILE_TYPE"]
        model_version = values["MODEL"]
        language = values["LANGUAGE"]

        if not audio_file or not output_folder or not output_name:
            sg.popup("Error", "Please fill in all fields.")
            continue

        try:
            model_output = model.transcribe(audio_file, language=language)
            text = model_output["text"]
        except Exception as e:
            sg.popup("Error", f"Transcription failed: {str(e)}")
            continue

        if output_file_type == "txt":
            try:
                output_file = f"{output_name}.txt"
                output_path = os.path.join(output_folder, output_file)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except:
                sg.popup("Error", f"Saving txt file failed: {str(e)}")
                continue
        elif output_file_type == "docx":
            try:
                output_file = f"{output_name}.docx"
                output_path = os.path.join(output_folder, output_file)
                document = Document()
                document.add_paragraph(text)
                document.save(output_path)
            except:
                sg.popup("Error", f"Saving docx file failed: {str(e)}")
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
