"""
configs.py -- paths used by every script in this folder.

Each path is read from an environment variable, so no machine-specific path is stored in the code:
    FEEDBACK_DATA_DIR      folder containing the patient .mat files (one file per session)    default: ./data
    FEEDBACK_IMAGES_DIR    where prediction / loss plots are written                          default: ./results/images
    FEEDBACK_RESULTS_DIR   where model weights, training histories and predictions are written default: ./results/models

Set them in your shell before running, e.g.
    export FEEDBACK_DATA_DIR=/path/to/your/data
or put them in a .env file in this folder (VS Code loads .env automatically when running Python files).
"""
import os

_repo_dir = os.path.dirname(os.path.abspath(__file__))

path_to_data = os.environ.get("FEEDBACK_DATA_DIR", os.path.join(_repo_dir, "data"))

path_to_save_models_images = os.environ.get("FEEDBACK_IMAGES_DIR", os.path.join(_repo_dir, "results", "images"))

path_to_save_models_results = os.environ.get("FEEDBACK_RESULTS_DIR", os.path.join(_repo_dir, "results", "models"))
