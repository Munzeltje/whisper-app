import os
import tempfile
import pytest
import configparser

from src.util import load_huggingface_token


def test_load_huggingface_token():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini") as tmp:
        tmp.write("[huggingface]\ntoken = test_token\n")
        tmp.flush()

        test_token = load_huggingface_token(tmp.name)
        expected_token = "test_token"

        assert test_token == expected_token


def test_load_huggingface_token_no_file():
    with pytest.raises(KeyError):
        load_huggingface_token("non_existent_file.ini")


def test_load_huggingface_token_no_token_in_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini") as tmp:
        tmp.write("[huggingface]\n")
        tmp.flush()

        with pytest.raises(KeyError):
            load_huggingface_token(tmp.name)

