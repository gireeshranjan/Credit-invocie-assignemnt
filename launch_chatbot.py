"""Launch the local portfolio chatbot; Ctrl+C stops the server."""
import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    root = Path(__file__).resolve().parent
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(root/"scripts"/"chatbot_app.py"),
                    "--server.address", "127.0.0.1", "--server.port", str(args.port),
                    "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                   cwd=root, check=True)
