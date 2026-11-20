"""Run the full pipeline in order and record the runtime of each step (results/runtime.csv).

Steps that need the internet (fetch_wdi.py) and the IFSB chart digitisation read their raw inputs from
data_raw/; run them first if data_raw/ is empty (see README.md).
"""
import os
import platform
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = ["digitize_ifsb.py", "validate_ifsb.py", "build_panel.py", "psifi_check.py", "a1_ceiling.py", "a2_barriers.py",
         "a3_usage.py", "a3b_gaps.py", "a4_annual.py", "a5_simulation.py"]


def main():
    rows = []
    for s in STEPS:
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "src", s)], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        dt = time.time() - t0
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-4000:])
            raise SystemExit(f"step failed: {s}")
        rows.append(dict(step=s, seconds=round(dt, 1)))
        print(f"{s:22s} {dt:7.1f} s")
    df = pd.DataFrame(rows)
    df.loc[len(df)] = dict(step="total", seconds=round(df.seconds.sum(), 1))
    df["machine"] = f"{platform.processor()} | {platform.system()} {platform.release()} | Python {platform.python_version()}"
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "runtime.csv"), index=False)


if __name__ == "__main__":
    main()
