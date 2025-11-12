#!/usr/bin/env python3
"""
Simple script to run the Weighted Voronoi Stippling algorithm.
Just edit the variables below and run: python main.py
"""

import os
import sys
import subprocess


def run_stippling_cmd():
    # =============================================================================
    # CONFIGURATION - Edit these variables to customize the algorithm
    # =============================================================================

    # Change this to point to your image file
    FILENAME = "data/original/plant4h.png"
    # Typical values: 25 (fast), 50 (good), 100 (high quality)
    N_ITERATIONS = 50
    # Typical values: 5000 (fast), 20000 (good), 50000 (high quality)
    N_POINTS = 20000
    # Controls the size of dots in the final image
    POINT_SIZE_MIN = 0.5
    POINT_SIZE_MAX = 2.5
    # Controls the output image size
    FIGSIZE = 6
    # Overwrite existing results
    FORCE = True    
    # Grey level threshold (0-255)
    THRESHOLD = 255
    # Save the stippled image to file
    SAVE = True          
    # Display final result in a window
    DISPLAY = False
    # Show progress during computation (slower)
    INTERACTIVE = False     

    # NOTE: Extra
    # Invert image colors (black becomes white, white becomes black) - Useful for images where you want to stipple the light areas instead of dark areas
    # INVERT_COLORS = False
    # Export overlay image with yellow points over grayscale background
    # OVERLAY_MODE = True

    # =============================================================================
    # SCRIPT EXECUTION - Don't modify below unless you know what you're doing
    # =============================================================================
                            
    """Run the stippling algorithm with the configured parameters."""
    
    # Check if image exists
    if not os.path.exists(FILENAME):
        print(f"Error: Image file '{FILENAME}' not found!")
        print("Please update the FILENAME variable in this script.")
        return 1
    
    # Path to the stippler script
    stippler_path = os.path.join("code", "stippler.py")
    # stippler_path = os.path.join("src", "stippler.py")
    
    if not os.path.exists(stippler_path):
        print(f"Error: Stippler script '{stippler_path}' not found!")
        print("Make sure you're running this script from the repository root.")
        return 1
    
    # Build command
    cmd = [
        sys.executable, stippler_path, FILENAME,
        "--n_iter", str(N_ITERATIONS),
        "--n_point", str(N_POINTS),
        "--pointsize", str(POINT_SIZE_MIN), str(POINT_SIZE_MAX),
        "--figsize", str(FIGSIZE),
        "--threshold", str(THRESHOLD)
    ]
    
    # Add optional flags
    if FORCE:
        cmd.append("--force")
    if SAVE:
        cmd.append("--save")
    if DISPLAY:
        cmd.append("--display")
    if INTERACTIVE:
        cmd.append("--interactive")
    # NOTE: Extra
    # if INVERT_COLORS:
    #     cmd.append("--invert")
    # if OVERLAY_MODE:
    #     cmd.append("--overlay")
    
    # Print configuration
    print("Weighted Voronoi Stippling")
    print("=" * 30)
    print(f"Filename: {FILENAME}")
    print(f"Points: {N_POINTS}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Point Size: ({POINT_SIZE_MIN}, {POINT_SIZE_MAX})")
    print(f"Figure Size: {FIGSIZE}")
    print(f"Threshold: {THRESHOLD}")
    print()
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run the command
    result = subprocess.run(cmd)
    return result.returncode


def run_stippling():
    import argparse
    SRC_PATH = os.path.join(os.path.dirname(__file__), "src")
    sys.path.append(SRC_PATH)
    from src.stippler import main as stippler_main
    
    args = argparse.ArgumentParser().parse_args()
    args.filename = "data/original/plant4h.png"
    args.n_iter = 5
    args.n_point = 5000
    args.pointsize = (1, 1)
    args.figsize = 6
    args.force = True
    args.threshold = 255
    args.save = True
    # args.display = False
    # args.interactive = False

    args.accelerator = "cuda"  # 'none', 'numpy', 'numba', 'cuda'
    args.invert = False
    # args.overlay = False

    # Check if image exists
    if not os.path.exists(args.filename):
        print(f"Error: Image file '{args.filename}' not found!")
        print("Please update the IMAGE_PATH variable in this script.")
        return 1

    stippler_main(args)


if __name__ == "__main__":
    # run_stippling_cmd()
    run_stippling()
