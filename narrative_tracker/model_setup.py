"""
model_setup.py
--------------
Ensures the spaCy en_core_web_sm model is available.

Called at app startup so Streamlit Cloud (which runs a fresh environment
each deploy) has the model downloaded before the first analysis request.

The download is cached after first run — subsequent imports are instant.
"""

import subprocess
import sys
import spacy


def ensure_spacy_model(model: str = "en_core_web_sm") -> None:
    """Download *model* if not already installed. Idempotent."""
    try:
        spacy.load(model)
    except OSError:
        print(f"Downloading spaCy model '{model}'…", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             f"https://github.com/explosion/spacy-models/releases/download/"
             f"{model}-3.8.0/{model}-3.8.0-py3-none-any.whl"],
            check=True,
            capture_output=True,
        )
        print(f"Model '{model}' installed.", flush=True)
