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
import tqdm
import os
import scipy.ndimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
DIR_PATH = os.path.dirname(__file__)
sys.path.append(DIR_PATH)

import voronoi


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
    image = Image.open(filename).convert(mode='L')  # Convert to grayscale

    # Resize image if requested
    if args.image_size is not None and image.size != args.image_size:
        image = image.resize(size=args.image_size, resample=Image.LANCZOS)

    if args.invert_image:
        image = Image.fromarray(255 - np.array(image))

    # Export to SOURCE_PATH
    image.save(args.source_filename)
    og_density = np.array(image, dtype=np.float32)

    # Invert image colors if requested
    if args.invert_density:
        og_density = 255.0 - og_density

    zoom = 1.0
    if args.zoom:
        # We want (approximately) 500 pixels per voronoi region
        zoom = (args.n_point * 500) / (og_density.shape[0] * og_density.shape[1])
        # Avoid zoom=0 for large images, which would create an empty density array.
        zoom = max(1, int(round(np.sqrt(zoom))))
        density = scipy.ndimage.zoom(og_density, zoom, order=0)
    else:
        density = og_density

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
    if args.zoom:
        H, W = og_density.shape[0], og_density.shape[1]
        scale_factor = 1.0 / zoom
        pts = np.rint(points * scale_factor).astype(int)

        # # DEBUG #
        # fig = plt.figure(figsize=(W/100, H/100), dpi=100,
        #                  facecolor="white")
        # ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        # ax.set_xlim([xmin, xmax])
        # ax.set_xticks([])
        # ax.set_ylim([ymin, ymax])
        # ax.set_yticks([])
        # scatter = ax.scatter(points[:, 0], points[:, 1], s=1, 
        #                      facecolor="k", edgecolor="None")
        # Pi = points.astype(int)
        # X = np.maximum(np.minimum(Pi[:, 0], og_density.shape[1]-1), 0)
        # Y = np.maximum(np.minimum(Pi[:, 1], og_density.shape[0]-1), 0)
        # sizes = (args.pointsize[0] +
        #          (args.pointsize[1]-args.pointsize[0])*og_density[Y, X])
        # scatter.set_offsets(points)
        # scatter.set_sizes(sizes)

        # # Save stipple points and stippled image
        # plt.savefig(args.target_filename)
        # plt.close(fig)
        # return
        # # DEBUG #
    
    else:
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
    data_path = r"/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3"
    # n = 10
    n = -1  # Set to -1 to process all images in the folder
    
    n_iter = 5
    n_point = 1024
    pointsize = (1, 1)
    # figsize = 6
    # force = True
    threshold = 255
    # display = False
    # interactive = False

    image_size = (512, 512)  # Width, Height
    accelerator = "numba"  # 'none', 'numpy', 'numba', 'cuda'
    invert_image = False
    invert_density = False
    zoom = True
    # overlay = False


    # NOTE: Define Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--n', type=int, default=n)
    parser.add_argument('--n_iter', type=int, default=n_iter)
    parser.add_argument('--n_point', type=int, default=n_point)
    parser.add_argument('--pointsize', type=int, nargs=2, default=pointsize)
    # parser.add_argument('--figsize', type=int, default=figsize)
    # parser.add_argument('--force', type=bool, default=force)
    parser.add_argument('--threshold', type=int, default=threshold)
    # parser.add_argument('--display',type=bool, default=display)
    # parser.add_argument('--interactive', type=bool, default=interactive)
    parser.add_argument('--image_size', type=int, nargs=2, default=image_size)
    parser.add_argument('--accelerator', type=str, default=accelerator)
    parser.add_argument('--invert_image', type=bool, default=invert_image)
    parser.add_argument('--invert_density', type=bool, default=invert_density)
    parser.add_argument('--zoom', type=bool, default=zoom)
    # parser.add_argument('--overlay', type=bool, default=overlay)
    args = parser.parse_args()


    # NOTE: Build paths
    IMAGES_PATH = os.path.join(args.data_path, "original")
    SOURCE_PATH = os.path.join(args.data_path, "source")
    TARGET_PATH = os.path.join(args.data_path, "target")
    JSON_PATH = os.path.join(args.data_path, "prompt.json")
    # OUTPUT_PATH = os.path.join(args.data_path, "output")
    # os.makedirs(OUTPUT_PATH, exist_ok=True)
    dataset_paths = dict(
        source_path=SOURCE_PATH,
        target_path=TARGET_PATH,
        json_path=JSON_PATH
    )
    os.makedirs(os.path.dirname(dataset_paths['json_path']), exist_ok=True)
    os.makedirs(dataset_paths['source_path'], exist_ok=True)
    os.makedirs(dataset_paths['target_path'], exist_ok=True)


    # NOTE: Aggregate image files
    image_files = sorted([
        os.path.relpath(os.path.join(root, f), IMAGES_PATH)
        for root, _, files in os.walk(IMAGES_PATH)
        for f in files
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))
    ])


    # NOTE: Generate images
    if args.n == -1:
        args.n = len(image_files)
    for i in tqdm.tqdm(range(args.n)):
        # if i < 1000:
        #     continue
        # if i > 13001:
        #     break
        args.filename = os.path.join(IMAGES_PATH, image_files[i])
        args.source_filename = os.path.join(SOURCE_PATH, image_files[i])
        args.target_filename = os.path.join(TARGET_PATH, image_files[i])
        os.makedirs(os.path.dirname(args.source_filename), exist_ok=True)
        os.makedirs(os.path.dirname(args.target_filename), exist_ok=True)
        run(args)


    # NOTE: Export json
    json_data = []
    for i in tqdm.tqdm(range(args.n)):
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
