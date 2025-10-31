#! /usr/bin/env python3
# -----------------------------------------------------------------------------
# Weighted Voronoi Stippler
# Copyright (2017) Nicolas P. Rougier - BSD license
#
# Implementation of:
#   Weighted Voronoi Stippling, Adrian Secord
#   Symposium on Non-Photorealistic Animation and Rendering (NPAR), 2002
# -----------------------------------------------------------------------------
import argparse
import sys
import os
DIR_PATH = os.path.dirname(__file__)
sys.path.append(DIR_PATH)

import voronoi
import numpy as np
from PIL import Image
from tqdm import tqdm


def normalize(D):
    Vmin, Vmax = D.min(), D.max()
    if Vmax - Vmin > 1e-5:
        D = (D-Vmin)/(Vmax-Vmin)
    else:
        D = np.zeros_like(D)
    return D


def initialization(n, D):
    """
    Return n points distributed over [xmin, xmax] x [ymin, ymax]
    according to (normalized) density distribution.

    with xmin, xmax = 0, density.shape[1]
         ymin, ymax = 0, density.shape[0]

    The algorithm here is a simple rejection sampling.
    """

    samples = []
    while len(samples) < n:
        # X = np.random.randint(0, D.shape[1], 10*n)
        # Y = np.random.randint(0, D.shape[0], 10*n)
        X = np.random.uniform(0, D.shape[1], 10*n)
        Y = np.random.uniform(0, D.shape[0], 10*n)
        P = np.random.uniform(0, 1, 10*n)
        index = 0
        while index < len(X) and len(samples) < n:
            x, y = X[index], Y[index]
            x_, y_ = int(np.floor(x)), int(np.floor(y))
            if P[index] < D[y_, x_]:
                samples.append([x, y])
            index += 1
    return np.array(samples)


def run(args):
    filename = args.filename
    image = Image.open(filename).convert('L')  # Convert to grayscale
    image = image.resize(args.image_size, Image.LANCZOS)

    # Export to SOURCE_PATH
    image.save(args.source_filename)

    density = np.array(image, dtype=np.float32)
    
    # Invert image colors if requested
    if args.invert:
        density = 255.0 - density

    # Apply threshold onto image
    # Any color > threshold will be white
    density = np.minimum(density, args.threshold)

    density = 1.0 - normalize(density)
    density = density[::-1, :]
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    # Initialization
    points = initialization(args.n_point, density)
        
    xmin, xmax = 0, density.shape[1]
    ymin, ymax = 0, density.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])

    # Non-interactive mode
    for i in range(args.n_iter):
        regions, points = voronoi.centroids(points, density, bbox, density_P, density_Q, accelerator=args.accelerator)

    # Export final result
    # Save binary PNG of stipple points (one-pixel dots)
    H, W = density.shape[0], density.shape[1]
    pts = np.rint(points).astype(int)
    # Clip to image bounds
    x = np.clip(pts[:, 0], 0, W - 1)
    y = np.clip(pts[:, 1], 0, H - 1)
    # Convert to top-left origin for image coordinates
    y_img = (H - 1) - y

    # # Create binary mask: white background (0), black points (255)
    # mask = np.zeros((H, W), dtype=np.uint8)
    # mask[y_img, x] = 255  # 255 = white points

    # Create binary mask: white background (255), black points (0)
    mask = np.full((H, W), 255, dtype=np.uint8)
    mask[y_img, x] = 0  # 0 = black points

    # Export to OUTPUT_PATH
    Image.fromarray(mask, mode='L').convert('1').save(args.target_filename)


# =============================================================================
def main():
    ############################
    # CONFIGURATION PARAMETERS #
    ############################
    ROOT_PATH = ""
    DATA_FOLDER = "data"

    SOURCE_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "source")
    TARGET_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "target")
    JSON_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "prompt.json")
    IMAGES_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "original")
    # OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
    # os.makedirs(OUTPUT_PATH, exist_ok=True)

    # N = 10
    N = -1  # Set to -1 to process all images in the folder

    dataset_paths = dict(
        source_path=SOURCE_PATH,
        target_path=TARGET_PATH,
        json_path=JSON_PATH
    )
    os.makedirs(os.path.dirname(dataset_paths['json_path']), exist_ok=True)
    os.makedirs(dataset_paths['source_path'], exist_ok=True)
    os.makedirs(dataset_paths['target_path'], exist_ok=True)

    args = argparse.ArgumentParser().parse_args()
    
    args.n_iter = 5
    args.n_point = 5000
    # args.pointsize = (1, 1)
    # args.figsize = 6
    # args.force = True
    args.threshold = 255
    # args.display = False
    # args.interactive = False

    args.image_size = (512, 512)  # Width, Height
    args.accelerator = "cuda"  # 'none', 'numpy', 'numba', 'cuda'
    args.invert = False
    # args.overlay = False

    image_files = sorted([f for f in os.listdir(IMAGES_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    # Generate images
    if N == -1:
        N = len(image_files)
    for i in tqdm(range(N)):
        args.filename = os.path.join(IMAGES_PATH, image_files[i])
        args.source_filename = os.path.join(SOURCE_PATH, image_files[i])
        args.target_filename = os.path.join(TARGET_PATH, image_files[i])
        run(args)

    # Export json
    json_data = []
    for i in tqdm(range(N)):
        json_data.append({
            "source": f"source/{image_files[i]}",
            "target": f"target/{image_files[i]}",
            "prompt": f"Stippling"
        })
    with open(JSON_PATH, 'w') as f:
        for item in json_data:
            data_line = str(item).replace("\'", "\"")
            f.write(f"{data_line}\n")
    

if __name__ == '__main__':
    main()
