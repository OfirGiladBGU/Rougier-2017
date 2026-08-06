#!/usr/bin/env python3
"""Run wvs_data_gen.py for all ICONS-TIMES-V2 n_point values.

After each run, the output subfolders (target, timestamps) are renamed
with a _WVS_<N_POINT> postfix. original/ and source/ are left untouched.

All run parameters are passed explicitly on the command line so this driver
does not depend on the defaults inside wvs_data_gen.py.
"""

import subprocess
import sys
from pathlib import Path

# -- Configuration -------------------------------------------------------------

SCRIPT_PATH = Path(__file__).parent / "wvs_data_gen.py"

DATA_PATH = Path(
    "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion"
    "/experiments/outputs/icons_results_runtimes_wvs"
)

# Full ICONS - TIMES - V2 parameter set (passed explicitly; do not assume the
# underlying script defaults).
N = -1                  # -1 == process all images
N_ITER = 50
IMAGE_SIZE = (512, 512)
POINTSIZE = (1, 1)
THRESHOLD = 255
ACCELERATOR = "numba"
INVERT_IMAGE = False
INVERT_DENSITY = False
ZOOM = True
TRACK_TIME = True

# ICONS - TIMES - V2  (actual point counts; comments show NxN equivalent)
N_POINTS = [
    256,    # 16
    576,    # 24
    1024,   # 32
    1600,   # 40
    2304,   # 48
    3136,   # 56
    4096,   # 64
    5184,   # 72
    6400,   # 80
    7744,   # 88
    9216,   # 96
    10816,  # 104
    12544,  # 112
]

# Subfolders produced by each run that should be renamed after completion
# original/ and source/ are left untouched
OUTPUT_SUBDIRS = ["target", "timestamps"]

# -- Main ----------------------------------------------------------------------

def main():
    if not SCRIPT_PATH.exists():
        print(f"ERROR: Script not found at {SCRIPT_PATH}")
        sys.exit(1)

    print(f"Running {len(N_POINTS)} n_point values: {N_POINTS}")
    print(f"Script: {SCRIPT_PATH}\n")

    for n_point in N_POINTS:
        print(f"\n{'='*70}")
        print(f"  N_POINT = {n_point}")
        print(f"{'='*70}\n")

        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--data_path", str(DATA_PATH),
            "--n", str(N),
            "--n_point", str(n_point),
            "--n_iter", str(N_ITER),
            "--image_size", str(IMAGE_SIZE[0]), str(IMAGE_SIZE[1]),
            "--pointsize", str(POINTSIZE[0]), str(POINTSIZE[1]),
            "--threshold", str(THRESHOLD),
            "--accelerator", ACCELERATOR,
            "--invert_image" if INVERT_IMAGE else "--no-invert_image",
            "--invert_density" if INVERT_DENSITY else "--no-invert_density",
            "--zoom" if ZOOM else "--no-zoom",
            "--track_time" if TRACK_TIME else "--no-track_time",
        ]
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nWarning: n_point={n_point} exited with code {result.returncode}")

        for subdir in OUTPUT_SUBDIRS:
            src = DATA_PATH / subdir
            dst = DATA_PATH / f"{subdir}_WVS_{n_point}"
            if src.exists():
                src.rename(dst)
                print(f"[rename] {subdir}  ->  {dst.name}")

    print(f"\n{'='*70}")
    print("All n_point runs complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
