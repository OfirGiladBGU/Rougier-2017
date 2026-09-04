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
import time
import scipy.ndimage
import numpy as np
from PIL import Image

try:
    import cv2  # only needed for --apply_preprocess
except Exception:
    cv2 = None
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


def points_to_canonical(points, width, height, scale=1.0):
    """WVS solver output -> canonical (N,2) float64, x-then-y, [0,1], y increasing DOWNWARD.

    Canonical is the convention control_v4/train_control.py:extract_points_from_target returns
    ([cx / w, cy / h]), so an exported .npy is a drop-in replacement for centroid detection.

    The relaxation runs in density-array pixel coordinates, and the density was row-flipped at load
    (density = density[::-1, :]), so y points UP there -- the same reason the rasteriser below
    computes y_img = (H - 1) - y. `scale` is the 1/zoom factor that maps zoomed-density pixels back
    into original-image pixels; `width`/`height` must be the frame the scaled points live in.

    This omits the rasteriser's np.rint(), which is the only lossy step. Note the two differ by up
    to half a pixel: the rasteriser treats the pixel index as rint(coordinate), while normalising by
    W/H is what agrees with extract_points_from_target's cx / w. The .npy follows the latter.
    """
    pts = np.asarray(points, dtype=np.float64) * float(scale)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] / float(width)
    out[:, 1] = 1.0 - pts[:, 1] / float(height)
    # Half-open [0, 1): a coordinate of exactly 1.0 indexes one past the last pixel downstream.
    return np.clip(out, 0.0, 1.0 - 1e-9)


def save_points_npy(points, out_path, n_expected=None):
    """Write canonical coordinates atomically.

    n_expected is NOT enforced. The solver writes whatever it produced; repair happens in
    exactly ONE place -- train_control._fit_points_to_n -- which duplicates an existing point
    to reach the grid budget. Duplication is the only repair that is REVERSIBLE: a duplicate
    has nearest-neighbour distance exactly 0, so it is trivially detectable and removable,
    and dropping it recovers the true statistics exactly. A uniform-random pad is
    indistinguishable from a real point and can never be undone.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points, got {pts.shape}")
    if n_expected is not None and len(pts) != n_expected:
        print(f"  [warn] wrote {len(pts)} points (expected {n_expected}): {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # np.save() APPENDS ".npy" when handed a path, which is why the temp name used to have to
    # end in that extension itself -- leaving interrupted runs behind a temp file that any *.npy
    # glob over the target dir would pick up as a real export. Passing a file handle suppresses
    # the append, so the temp is a plain "<stem>.npy.tmp" and cannot be mistaken for one.
    tmp = str(out_path) + ".tmp"
    with open(tmp, "wb") as handle:
        np.save(handle, pts)
    os.replace(tmp, str(out_path))


# --- GBN preprocessing (ported from GaussianBlueNoise/scripts/image_preprocess.py) ---
def percentile_stretch(gray, p_low=1.0, p_high=99.0):
    lo, hi = np.percentile(gray, [p_low, p_high])
    if hi - lo < 1e-6:
        return gray.copy()
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(gray, clip_limit=3.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def unsharp_mask(gray, sigma=1.4, amount=1.5):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray.astype(np.float32), 1.0 + amount, blur.astype(np.float32), -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def suppress_background(gray):
    g = gray.copy()
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    h, w = bin_img.shape
    flood = bin_img.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)
    bg = flood == 128
    out = g.astype(np.float32)
    out[bg] = 0.30 * out[bg] + 0.70 * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_image(gray, do_bg_suppression=True):
    """GBN enhancement: stretch -> CLAHE -> unsharp -> stretch -> optional bg-suppress."""
    if cv2 is None:
        raise RuntimeError("cv2 (opencv-python) is required for --apply_preprocess")
    x = percentile_stretch(gray, 1.0, 99.0)
    x = apply_clahe(x, 3.0, 8)
    x = unsharp_mask(x, 1.4, 1.5)
    x = percentile_stretch(x, 0.8, 99.2)
    if do_bg_suppression:
        x = suppress_background(x)
    return x


def load_gray_on_white(filename):
    """Open as grayscale, compositing any transparency onto WHITE so a transparent
    background becomes white. Plain .convert('L') on an RGBA image ignores alpha and
    turns transparent (RGB 0,0,0) pixels BLACK -- which the stippler then fills with dots.
    """
    img = Image.open(filename)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def run(args):
    filename = args.filename
    image = load_gray_on_white(filename)  # transparency -> white, then grayscale

    # Resize image if requested
    if args.image_size is not None and image.size != args.image_size:
        image = image.resize(size=args.image_size, resample=Image.LANCZOS)

    # Optional GBN-style grayscale enhancement, before invert/save (matches gbn_data_gen).
    if args.apply_preprocess:
        gray = preprocess_image(np.array(image), do_bg_suppression=not args.disable_bg_suppression)
        image = Image.fromarray(gray)

    if args.invert_image:
        image = Image.fromarray(255 - np.array(image))

    # Export to SOURCE_PATH. When the input IS the source (dataset has no original/), this
    # would overwrite the file we just read -- skip it and leave source/ exactly as staged.
    if os.path.abspath(filename) != os.path.abspath(args.source_filename):
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
    # Resolve the output frame ONCE, before choosing a representation: with --zoom the relaxation
    # runs on the zoomed density, so points must be scaled back into original-image pixels.
    if args.zoom:
        H, W = og_density.shape[0], og_density.shape[1]
        scale_factor = 1.0 / zoom

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
        scale_factor = 1.0

    # NPY: exact continuous coordinates, canonical [0,1] x-then-y with y DOWN. Same frame and the
    # same y-flip the mask below applies (the density was row-flipped at load), but without the
    # np.rint() -- the rounding is the only lossy step, and it is what merges points that land in
    # one pixel.
    if args.export_npy:
        npy_filename = os.path.splitext(args.target_filename)[0] + ".npy"
        save_points_npy(
            points_to_canonical(points, width=W, height=H, scale=scale_factor),
            npy_filename,
            n_expected=args.n_point,
        )

    # Save binary PNG of stipple points (one-pixel dots)
    if args.export_png:
        pts = np.rint(points * scale_factor).astype(int)

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
    # Defaults #
    n = -1  # Set to -1 to process all images in the folder
    # n = 10
    
    n_iter = 50
    n_point = 1024
    pointsize = (1, 1)
    # figsize = 6
    # force = True
    threshold = 255
    # display = False
    # interactive = False

    # image_size = (512, 512)  # Width, Height
    image_size = None  # Keep original size
    accelerator = "numba"  # 'none', 'numpy', 'numba', 'cuda'
    invert_image = False
    invert_density = False
    apply_preprocess = False
    disable_bg_suppression = False
    zoom = True
    export_png = True  # Write the rasterised target .png
    export_npy = True  # Write exact continuous coordinates as target .npy
    overwrite = False  # Re-run images whose target outputs already exist
    # overlay = False
    track_time = True

    target_folder = "target"

    ############################
    # CONFIGURATION PARAMETERS #
    ############################

    # Icons-50 - dataset
    # data_path = r"/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_WVS"
    # n_point = 1024
    # image_size = (512, 512)
    # track_time = False

    # CelebA-5K - dataset
    # data_path = r"/groups/asharf_group/ofirgila/ControlNet/training/CelebA-5K_1024_WVS"
    # n_point = 1024
    # image_size = (512, 512)
    # apply_preprocess = True
    # track_time = False

    # ShapeNetRendering - dataset
    # data_path = r"/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRendering-3K_256_WVS"
    # n_point = 256
    # image_size = None
    # track_time = False

    # ShapeNetRenderingV2 - dataset
    # data_path = r"/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRenderingV2-3K_576_WVS"
    # n_point = 576
    # image_size = (448, 448)
    # apply_preprocess = True
    # track_time = False

    # AirplaneCarShip - dataset
    # data_path = r"/groups/asharf_group/ofirgila/ControlNet/training/AirplaneCarShip-3K_1600_WVS"
    # n_point = 1600
    # image_size = (384, 384)
    # apply_preprocess = True
    # track_time = False

    # ShapeNetRender_Custom - dataset
    # data_path = r"/groups/asharf_group/ofirgila/ControlNet/training/ShapeNetRender_Custom-3K_1600_WVS"
    # n_point = 1600
    # image_size = None
    # apply_preprocess = True
    # track_time = False


    # Quadratic Sample
    # data_path = r"/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/images_results_metrics/quadratic"
    # n_point = 1024
    # image_size = None
    # track_time = False
    # target_folder = f"target_WVS_{n_point}"

    # Monkey Sample
    # data_path = r"/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/images_results_metrics/monkey"
    # n_point = 1024
    # image_size = None
    # track_time = False
    # target_folder = f"target_WVS_{n_point}"

    # Plant Sample
    # data_path = r"/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/images_results_metrics/plant2"
    # n_point = 1024
    # image_size = None
    # track_time = False
    # target_folder = f"target_WVS_{n_point}"


    # Spectral Analysis Set Sample
    data_path = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/spectral_analysis"
    n_point = 1024
    apply_preprocess = False
    image_size = None
    track_time = False
    target_folder = f"target_WVS_{n_point}"


    # Faces Set Sample
    # data_path = r"/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/faces_results_compare"
    # n_point = 1024
    # image_size = (512, 512)
    # track_time = False
    # target_folder = f"target_WVS_{n_point}"

    # Icons-50 - METRICS
    # data_path = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/quantitative_advance_metrics"
    # n_point = 1024
    # track_time = False
    # target_folder = f"target_WVS_{n_point}"

    # Icons-50 - TIMES - V1
    # data_path = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/icons_results_runtimes"
    # n_point = 576
    # n_point = 1024
    # n_point = 2304
    # image_size = (512, 512)
    # target_folder = f"target_WVS_{n_point}"


    # NOTE: Define Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--n', type=int, default=n)
    parser.add_argument('--n_iter', type=int, default=n_iter)
    parser.add_argument('--n_point', type=int, default=n_point)
    parser.add_argument('--pointsize', type=int, nargs=2, default=pointsize)
    # parser.add_argument('--figsize', type=int, default=figsize)
    # parser.add_argument('--force', action=argparse.BooleanOptionalAction, default=force)
    parser.add_argument('--threshold', type=int, default=threshold)
    parser.add_argument('--target_folder', type=str, default=target_folder,
                        help='Output folder name under --data_path (e.g. target_WVS_1024).')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=overwrite,
                        help='Re-run images whose target outputs already exist. '
                             '--no-overwrite skips them, to resume a partial run.')
    # parser.add_argument('--display', action=argparse.BooleanOptionalAction, default=display)
    # parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, default=interactive)
    parser.add_argument('--image_size', type=int, nargs=2, default=image_size)
    parser.add_argument('--accelerator', type=str, default=accelerator)
    parser.add_argument('--invert_image', action=argparse.BooleanOptionalAction, default=invert_image)
    parser.add_argument('--invert_density', action=argparse.BooleanOptionalAction, default=invert_density)
    parser.add_argument('--apply_preprocess', action=argparse.BooleanOptionalAction, default=apply_preprocess,
                        help='Apply the GBN preprocessing pipeline to the source before stippling')
    parser.add_argument('--disable_bg_suppression', action=argparse.BooleanOptionalAction, default=disable_bg_suppression,
                        help='Skip the background-suppression step of the preprocess')
    parser.add_argument('--zoom', action=argparse.BooleanOptionalAction, default=zoom)
    parser.add_argument('--export_png', action=argparse.BooleanOptionalAction, default=export_png,
                        help="Write the rasterised target .png")
    parser.add_argument('--export_npy', action=argparse.BooleanOptionalAction, default=export_npy,
                        help="Write exact continuous coordinates as target .npy")
    parser.add_argument('--track_time', action=argparse.BooleanOptionalAction, default=track_time,
                        help="Enable time tracking; saves elapsed time per image to 'timestamps/' subfolder")
    # parser.add_argument('--overlay', action=argparse.BooleanOptionalAction, default=overlay)
    args = parser.parse_args()


    # NOTE: Build paths
    # Read inputs from original/ when present, else source/. Reading from source/ means the
    # images ARE the source: there is no original -> source step to perform, so preprocessing
    # is forced off (it would preprocess an already-preprocessed image) and run() leaves
    # source/ untouched.
    ORIGINAL_PATH = os.path.join(args.data_path, "original")
    SOURCE_PATH = os.path.join(args.data_path, "source")
    use_original = os.path.isdir(ORIGINAL_PATH)
    IMAGES_PATH = ORIGINAL_PATH if use_original else SOURCE_PATH
    if not os.path.isdir(IMAGES_PATH):
        raise FileNotFoundError(
            f"No 'original/' or 'source/' folder under: {args.data_path}")
    if not use_original and args.apply_preprocess:
        print("Note: no 'original/' folder -- reading from 'source/' and skipping "
              "--apply_preprocess (those images are already the source).", file=sys.stderr)
        args.apply_preprocess = False
    TARGET_PATH = os.path.join(args.data_path, args.target_folder)
    JSON_PATH = os.path.join(args.data_path, "prompt.json")
    TIMESTAMPS_PATH = os.path.join(args.data_path, "timestamps") if args.track_time else None
    dataset_paths = dict(
        source_path=SOURCE_PATH,
        target_path=TARGET_PATH,
        json_path=JSON_PATH
    )
    os.makedirs(os.path.dirname(dataset_paths['json_path']), exist_ok=True)
    os.makedirs(dataset_paths['source_path'], exist_ok=True)
    os.makedirs(dataset_paths['target_path'], exist_ok=True)
    if args.track_time:
        os.makedirs(TIMESTAMPS_PATH, exist_ok=True)


    # NOTE: Aggregate image files
    image_files = sorted([
        os.path.relpath(os.path.join(root, f), IMAGES_PATH)
        for root, _, files in os.walk(IMAGES_PATH)
        for f in files
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))
    ])
    if not image_files:
        raise FileNotFoundError(f"No images found under: {IMAGES_PATH}")


    # NOTE: Generate images
    if args.n == -1:
        args.n = len(image_files)
    args.n = min(args.n, len(image_files))
    skipped = 0
    for i in tqdm.tqdm(range(args.n)):
        args.filename = os.path.join(IMAGES_PATH, image_files[i])
        # Always write source/target as PNG (convert .jpg etc.), matching gbn_data_gen.
        rel_png = os.path.splitext(image_files[i])[0] + ".png"
        args.source_filename = os.path.join(SOURCE_PATH, rel_png)
        args.target_filename = os.path.join(TARGET_PATH, rel_png)
        os.makedirs(os.path.dirname(args.source_filename), exist_ok=True)
        os.makedirs(os.path.dirname(args.target_filename), exist_ok=True)

        # Resume support. Only the outputs THIS run would write are checked, so
        # flipping an export flag cannot make it skip work it has not done.
        if not args.overwrite:
            npy_path = os.path.splitext(args.target_filename)[0] + '.npy'
            ready = ((os.path.exists(args.target_filename) if args.export_png else True)
                     and (os.path.exists(npy_path) if args.export_npy else True))
            if ready:
                skipped += 1
                continue
        
        # Time tracking
        if args.track_time:
            start_time = time.time()
        
        run(args)
        
        # Save timing info if tracking is enabled
        if args.track_time:
            elapsed = time.time() - start_time
            # Build the timestamp file path (same relative structure)
            rel_path = image_files[i]
            timestamp_file_path = os.path.join(TIMESTAMPS_PATH, os.path.splitext(rel_path)[0] + ".txt")
            os.makedirs(os.path.dirname(timestamp_file_path), exist_ok=True)
            with open(timestamp_file_path, 'w') as f:
                f.write(f"{elapsed:.6f}\n")


    # NOTE: Export json
    json_data = []
    for i in tqdm.tqdm(range(args.n)):
        json_data.append({
            "source": f"source/{os.path.splitext(image_files[i])[0]}.png",
            "target": f"target/{os.path.splitext(image_files[i])[0]}.png",
            "prompt": f"Stippling"
        })
    with open(JSON_PATH, 'w') as f:
        for item in json_data:
            data_line = str(item).replace("\'", "\"")
            f.write(f"{data_line}\n")
    

if __name__ == '__main__':
    main()
