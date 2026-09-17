"""Root runner for full reproduction pipeline (scripts 01-04)."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    projectRoot = Path(__file__).resolve().parent
    scriptPath = projectRoot / "scripts" / "run_all.py"
    ret = subprocess.run([sys.executable, str(scriptPath)], cwd=str(projectRoot))
    sys.exit(ret.returncode)
