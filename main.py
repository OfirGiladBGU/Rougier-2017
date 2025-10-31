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

    # INPUT IMAGE PATH
    # Change this to point to your image file
    IMAGE_PATH = "data/original/plant4h.png"

    # NUMBER OF STIPPLE POINTS
    # More points = better detail but slower processing
    # Typical values: 5000 (fast), 20000 (good), 50000 (high quality)
    N_POINTS = 20000

    # NUMBER OF ITERATIONS  
    # More iterations = better convergence but slower processing
    # Typical values: 25 (fast), 50 (good), 100 (high quality)
    N_ITERATIONS = 25

    # POINT SIZE (min, max)
    # Controls the size of dots in the final image
    # Smaller values = finer detail
    POINT_SIZE_MIN = 1.5
    POINT_SIZE_MAX = 1.5

    # FIGURE SIZE
    # Controls the output image size
    FIGURE_SIZE = 6

    # THRESHOLD
    # Grey level threshold (0-255)
    # Higher values = more selective about dark areas
    THRESHOLD = 255

    # OUTPUT OPTIONS
    SAVE_RESULT = True          # Save the stippled image to file
    FORCE_RECOMPUTE = True      # Overwrite existing results
    SHOW_INTERACTIVE = True     # Show progress during computation (slower)
    SHOW_FINAL = False          # Display final result in a window

    # IMAGE PROCESSING OPTIONS
    INVERT_COLORS = False       # Invert image colors (black becomes white, white becomes black)
                                # Useful for images where you want to stipple the light areas instead of dark areas

    # OVERLAY MODE
    OVERLAY_MODE = True         # Export overlay image with yellow points over grayscale background
                                # Output: *-stipple-overlay.png

    # =============================================================================
    # SCRIPT EXECUTION - Don't modify below unless you know what you're doing
    # =============================================================================
                            
    """Run the stippling algorithm with the configured parameters."""
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file '{IMAGE_PATH}' not found!")
        print("Please update the IMAGE_PATH variable in this script.")
        return 1
    
    # Path to the stippler script
    stippler_path = os.path.join("code", "stippler.py")
    
    if not os.path.exists(stippler_path):
        print(f"Error: Stippler script '{stippler_path}' not found!")
        print("Make sure you're running this script from the repository root.")
        return 1
    
    # Build command
    cmd = [
        sys.executable, stippler_path, IMAGE_PATH,
        "--n_point", str(N_POINTS),
        "--n_iter", str(N_ITERATIONS),
        "--pointsize", str(POINT_SIZE_MIN), str(POINT_SIZE_MAX),
        "--figsize", str(FIGURE_SIZE),
        "--threshold", str(THRESHOLD)
    ]
    
    # Add optional flags
    if SAVE_RESULT:
        cmd.append("--save")
    if FORCE_RECOMPUTE:
        cmd.append("--force")
    if SHOW_INTERACTIVE:
        cmd.append("--interactive")
    if SHOW_FINAL:
        cmd.append("--display")
    if INVERT_COLORS:
        cmd.append("--invert")
    if OVERLAY_MODE:
        cmd.append("--overlay")
    
    # Print configuration
    print("Weighted Voronoi Stippling")
    print("=" * 30)
    print(f"Image: {IMAGE_PATH}")
    print(f"Points: {N_POINTS}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Point size: {POINT_SIZE_MIN} - {POINT_SIZE_MAX}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Invert colors: {INVERT_COLORS}")
    print(f"Save result: {SAVE_RESULT}")
    print(f"Show progress: {SHOW_INTERACTIVE}")
    print(f"Overlay mode: {OVERLAY_MODE}")
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
    args.display = False
    args.interactive = False

    args.accelerator = "numba"
    args.invert = False
    args.overlay = False

    # Check if image exists
    if not os.path.exists(args.filename):
        print(f"Error: Image file '{args.filename}' not found!")
        print("Please update the IMAGE_PATH variable in this script.")
        return 1

    stippler_main(args)


if __name__ == "__main__":
    # sys.exit(run_stippling_cmd())
    run_stippling()
