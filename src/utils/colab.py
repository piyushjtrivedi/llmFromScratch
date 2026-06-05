import os
import sys
import logging

logger = logging.getLogger(__name__)

# Project folder name created inside Google Drive root.
_DRIVE_PROJECT_DIR = "llmFromScratch"


def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def get_data_dir() -> str:
    """
    Return the base data directory for this project.

    - Local / non-Colab: returns "data" (relative to the working directory).
    - Google Colab with Drive mounted: returns
      "/content/drive/MyDrive/llmFromScratch", creating the folder if needed.
    - Google Colab with Drive NOT yet mounted: attempts to mount it
      interactively, then returns the Drive path.
    - Google Colab but Drive mount fails: warns and falls back to the
      ephemeral "/content/data" path. Weights saved there are lost on
      runtime disconnect, so the warning is important.

    All callers (registry, weight loaders) call this at the moment they need
    a path, not at import time, so the Drive auth prompt is only triggered
    when weights are actually about to be read or written.
    """
    if not is_colab():
        return "data"

    drive_root = "/content/drive/MyDrive"
    project_path = os.path.join(drive_root, _DRIVE_PROJECT_DIR)

    if os.path.isdir(drive_root):
        os.makedirs(project_path, exist_ok=True)
        logger.info(f"[Colab] Using Google Drive path: {project_path}")
        return project_path

    # Drive not yet mounted — attempt interactive mount.
    logger.info("[Colab] Google Drive not mounted. Mounting now...")
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(project_path, exist_ok=True)
        logger.info(f"[Colab] Drive mounted. Using path: {project_path}")
        return project_path
    except Exception as e:
        logger.warning(
            "[Colab] Could not mount Google Drive — weights will be saved to "
            "ephemeral local storage and LOST on runtime disconnect.\n"
            "To persist weights, run this in a cell before training:\n"
            "  from google.colab import drive; drive.mount('/content/drive')\n"
            f"  (Error: {e})"
        )
        fallback = "/content/data"
        os.makedirs(fallback, exist_ok=True)
        return fallback
