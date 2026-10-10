import os
import subprocess
import sys

from init import ROOT
from train import IMPLEMENTATIONS, OOM_EXIT_CODE

CONFIG = {"steps": 50, "seq-len": 512, "seed": 42, "lr": 1e-4}
CSV = ROOT / "results/results.csv"


def main():
    for batch_size in [16, 24, 32]:
        CONFIG["batch-size"] = batch_size
        options = [item for key, value in CONFIG.items() for item in (f"--{key}", str(value))]
        environment = {**os.environ, "PYTHONHASHSEED": str(CONFIG["seed"])}
        for impl in IMPLEMENTATIONS:
            print(f"Running {impl}", flush=True)
            result = subprocess.run(
                    [sys.executable, str(ROOT / "train.py"), "--impl", impl,
                     "--csv", str(CSV), *options],
                env=environment, check=False,
            )
            if result.returncode == OOM_EXIT_CODE:
                print(f"Skipped {impl}: CUDA OOM", flush=True)
                continue
            if result.returncode:
                sys.exit(result.returncode)
    print(f"Results: {CSV}")


if __name__ == "__main__":
    main()
