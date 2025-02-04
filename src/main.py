import PySimpleGUI as sg

from app import create_layout, run_app
from util import load_huggingface_token


def main():
    hf_token = load_huggingface_token("config.ini")
    layout = create_layout()
    window = sg.Window("Whisper App", layout)
    run_app(hf_token, window)


if __name__ == "__main__":
    main()
