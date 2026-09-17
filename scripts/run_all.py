"""Runs scripts 01 through 04 end-to-end."""

import subprocess
import sys
import time
from pathlib import Path

projectRoot = Path(__file__).resolve().parent.parent

pipelineScripts = [
    ("scripts/01_visualize_neurobiology.py", "Neurobiology & Scalp Topographies"),
    ("scripts/02_analyze_manifold_geometry.py", "Manifold Geometry & RSA"),
    ("scripts/03_run_decoding_benchmark.py", "Decoding Benchmark & Feature Importance"),
    ("scripts/04_generate_synthetic_eeg.py", "Conditional Synthetic EEG Generation"),
]


def main():
    print("Running EEG affect reproduction pipeline (scripts 01-04)...")

    totalStart = time.time()
    for relPath, desc in pipelineScripts:
        scriptPath = projectRoot / relPath
        print(f"\n>>> Running {desc} ({relPath})...")
        t0 = time.time()
        ret = subprocess.run([sys.executable, str(scriptPath)], cwd=str(projectRoot))
        elapsed = time.time() - t0

        if ret.returncode != 0:
            print(f"[ERROR] {desc} failed with return code {ret.returncode}!")
            sys.exit(ret.returncode)
        print(f"[OK] Completed {desc} in {elapsed:.1f}s.")

    totalElapsed = time.time() - totalStart
    print(f"\nAll scripts completed successfully in {totalElapsed:.1f}s.")
    print(f"Figures and metrics saved to: {projectRoot / 'figures'}")


if __name__ == "__main__":
    main()
