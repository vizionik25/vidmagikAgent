"""Entry point — launches the AI Shorts Creator web app."""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the AI Shorts Creator web app."""
    app_main = Path(__file__).parent / "app" / "main.py"
    subprocess.run([sys.executable, str(app_main)])


if __name__ == "__main__":
    main()
