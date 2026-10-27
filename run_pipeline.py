"""Run data preparation, analytics, modelling and scoring from any working directory."""
import argparse
from pathlib import Path
import subprocess
import sys
from scripts.portfolio_common import ROOT, EXCEL

def run(skip_prepare=False):
    scripts = ([] if skip_prepare else ["prepare_data.py"]) + ["analytics.py", "train_models.py", "score_loans.py", "verify_outputs.py"]
    for script in scripts:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    print(f"Complete. Open the workbooks in {EXCEL}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-prepare", action="store_true", help="Use the existing analytics_base.csv.")
    run(parser.parse_args().skip_prepare)
