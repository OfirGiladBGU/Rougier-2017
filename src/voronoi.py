# -----------------------------------------------------------------------------
# Weighted Voronoi Stippler
# Copyright (2017) Nicolas P. Rougier - BSD license
# -----------------------------------------------------------------------------
import numpy as np
import scipy.spatial


# ----------------------------
# Utilities for acceleration modes
# ----------------------------

# Optional acceleration with Numba (CPU JIT)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# Optional GPU acceleration with PyTorch/CUDA
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

# Simple GPU cache for density cumulative arrays to avoid re-transfers each iteration
_GPU_CACHE = {
    'key': None,
    'P': None,
    'Q': None,
    'device': None,
}

def _get_PQ_gpu(P_np, Q_np, device='cuda'):
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    key = (id(P_np), id(Q_np), device)
    if _GPU_CACHE['key'] == key and _GPU_CACHE['P'] is not None and _GPU_CACHE['Q'] is not None:
        return _GPU_CACHE['P'], _GPU_CACHE['Q']
    # Transfer to GPU (float32 for performance)
    with torch.no_grad():
        P_gpu = torch.as_tensor(P_np, dtype=torch.float32, device=device)
        Q_gpu = torch.as_tensor(Q_np, dtype=torch.float32, device=device)
    _GPU_CACHE['key'] = key
    _GPU_CACHE['P'] = P_gpu
    _GPU_CACHE['Q'] = Q_gpu
    _GPU_CACHE['device'] = device
    return P_gpu, Q_gpu


def _build_outlines(vertices_list):
    """
    Build a single outline array and region id array for a list of regions.

    Returns
    -------
    O_all : (M,3) int array of [x1, x2, y]
    rid_all : (M,) int array region indices for each row in O_all
    n_regions : int number of regions
    """
    outlines = []
    region_ids = []
    n_regions = len(vertices_list)
    for idx, V in enumerate(vertices_list):
        if NUMBA_AVAILABLE:
            Vc = np.ascontiguousarray(V, dtype=np.float64)
            O = _rasterize_outline_numba(Vc)
        else:
            O = rasterize_outline(V)
        if O is None or len(O) == 0:
            continue
        outlines.append(O)
        region_ids.append(np.full((O.shape[0],), idx, dtype=np.int64))
    if len(outlines) == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64), n_regions
    O_all = np.vstack(outlines)
    rid_all = np.concatenate(region_ids)
    return O_all, rid_all, n_regions


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _rasterize_outline_numba(V):
        n = V.shape[0]
        X = V[:, 0]
        Y = V[:, 1]
        ymin = int(np.ceil(Y.min()))
        ymax = int(np.floor(Y.max()))
        if ymax < ymin:
            # Degenerate polygon
            return np.zeros((0, 3), dtype=np.int64)

        # Preallocate: at most 2 entries per scanline
        points = np.zeros((2 + (ymax - ymin) * 2, 3), dtype=np.int64)
        index = 0

        # Temporary storage for segment intersections (max n per scanline)
        seg = np.empty(n, dtype=np.float64)

        for y in range(ymin, ymax + 1):
            m = 0
            for i in range(n):
                index1 = (i - 1) % n
                index2 = i
                y1 = Y[index1]
                y2 = Y[index2]
                x1 = X[index1]
                x2 = X[index2]
                if y1 > y2:
                    y1, y2 = y2, y1
                    x1, x2 = x2, x1
                elif y1 == y2:
                    continue
                if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                    seg[m] = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    m += 1
            if m == 0:
                continue
            # Sort active intersections
            sorted_seg = np.sort(seg[:m])
            # Emit pairs
            # Ensure even pairing
            lim = (m // 2) * 2
            i2 = 0
            while i2 < lim:
                x1 = int(np.ceil(sorted_seg[i2]))
                x2 = int(np.ceil(sorted_seg[i2 + 1]))
                points[index, 0] = x1
                points[index, 1] = x2
                points[index, 2] = y
                index += 1
                i2 += 2
        return points[:index]


# ----------------------------
# CUDA/GPU-accelerated variants
# ----------------------------
if CUDA_AVAILABLE:
    def _weighted_centroids_batch_gpu(vertices_list, P_np, Q_np, device='cuda'):
        """
        Compute weighted centroids for multiple polygons in parallel on GPU.

        Strategy:
        - Rasterize region outlines on CPU (Numba if available) to get scanline segments
        - Concatenate all segments from all regions into a single tensor with region ids
        - Do one big GPU gather+reduction using scatter_add to accumulate per-region sums
        This minimizes CPU<->GPU transfers and per-region kernel overhead.
        """
        # Build outlines for all regions on CPU (Numba if available)
        O_all, rid_all, n_regions = _build_outlines(vertices_list)
        if O_all.shape[0] == 0:
            return np.zeros((n_regions, 2), dtype=np.float32)

        # Get (or build) GPU copies of P and Q once
        P_gpu, Q_gpu = _get_PQ_gpu(P_np, Q_np, device=device)
        height, width = P_gpu.shape

        with torch.no_grad():
            # Transfer outlines to GPU (int64), compute all at once
            y_vals = torch.as_tensor(O_all[:, 2], dtype=torch.int64, device=device)
            x1_vals = torch.as_tensor(O_all[:, 0], dtype=torch.int64, device=device)
            x2_vals = torch.as_tensor(O_all[:, 1], dtype=torch.int64, device=device)
            rids = torch.as_tensor(rid_all, dtype=torch.int64, device=device)

            # Clamp to valid ranges
            y_vals = torch.clamp(y_vals, 0, int(height) - 1)
            x1_vals = torch.clamp(x1_vals, 0, int(width) - 1)
            x2_vals = torch.clamp(x2_vals, 0, int(width) - 1)

            # Gather P and Q values (vectorized)
            P_y_x2 = P_gpu[y_vals, x2_vals]
            P_y_x1 = P_gpu[y_vals, x1_vals]
            Q_y_x2 = Q_gpu[y_vals, x2_vals]
            Q_y_x1 = Q_gpu[y_vals, x1_vals]

            # Compute per-segment contributions
            d_line = P_y_x2 - P_y_x1  # (M,)
            x_contrib = ((x2_vals.to(torch.float32) * P_y_x2 - Q_y_x2) -
                         (x1_vals.to(torch.float32) * P_y_x1 - Q_y_x1))
            y_contrib = y_vals.to(torch.float32) * d_line

            # Reduce per region using scatter_add
            d = torch.zeros(n_regions, dtype=torch.float32, device=device)
            x_sum = torch.zeros_like(d)
            y_sum = torch.zeros_like(d)
            d.scatter_add_(0, rids, d_line.to(torch.float32))
            x_sum.scatter_add_(0, rids, x_contrib)
            y_sum.scatter_add_(0, rids, y_contrib)

            # Avoid division by zero
            nonzero = d != 0
            cx = torch.zeros_like(d)
            cy = torch.zeros_like(d)
            cx[nonzero] = x_sum[nonzero] / d[nonzero]
            cy[nonzero] = y_sum[nonzero] / d[nonzero]

            # For zero-mass regions, fallback to zeros (rare)
            centroids = torch.stack([cx, cy], dim=1).cpu().numpy()

        return centroids


# ----------------------------
# Numba-accelerated variants
# ----------------------------
if NUMBA_AVAILABLE:
    def _weighted_centroids_batch_numba(vertices_list, P, Q):
        O_all, rid_all, n_regions = _build_outlines(vertices_list)
        if O_all.shape[0] == 0:
            return np.zeros((n_regions, 2), dtype=np.float32)
        # Ensure types compatible with numba
        O_all = np.ascontiguousarray(O_all, dtype=np.int64)
        rid_all = np.ascontiguousarray(rid_all, dtype=np.int64)
        P = np.ascontiguousarray(P)
        Q = np.ascontiguousarray(Q)
        C = _reduce_segments_numba(O_all, rid_all, P, Q, n_regions)
        return C.astype(np.float32)
    
    @njit(cache=True)
    def _reduce_segments_numba(O_all, rid_all, P, Q, n_regions):
        height = P.shape[0]
        width = P.shape[1]
        d = np.zeros(n_regions, dtype=np.float64)
        x_sum = np.zeros(n_regions, dtype=np.float64)
        y_sum = np.zeros(n_regions, dtype=np.float64)
        M = O_all.shape[0]
        for i in range(M):
            y = O_all[i, 2]
            if y < 0:
                continue
            if y >= height:
                y = height - 1
            x1 = O_all[i, 0]
            x2 = O_all[i, 1]
            if x1 < 0:
                x1 = 0
            if x1 >= width:
                x1 = width - 1
            if x2 < 0:
                x2 = 0
            if x2 >= width:
                x2 = width - 1

            p2 = P[y, x2]
            p1 = P[y, x1]
            d_line = p2 - p1
            rid = rid_all[i]
            d[rid] += d_line
            x_sum[rid] += (x2 * p2 - Q[y, x2]) - (x1 * p1 - Q[y, x1])
            y_sum[rid] += y * d_line

        centroids = np.zeros((n_regions, 2), dtype=np.float64)
        for r in range(n_regions):
            if d[r] != 0.0:
                centroids[r, 0] = x_sum[r] / d[r]
                centroids[r, 1] = y_sum[r] / d[r]
            else:
                centroids[r, 0] = 0.0
                centroids[r, 1] = 0.0
        return centroids


# ----------------------------
# Numpy-accelerated variants
# ----------------------------
def _weighted_centroids_batch_numpy(vertices_list, P, Q):
    """
    Batched centroid computation using NumPy vectorization and bincount.
    """
    O_all, rid_all, n_regions = _build_outlines(vertices_list)
    if O_all.shape[0] == 0:
        return np.zeros((n_regions, 2), dtype=np.float32)

    height, width = P.shape
    # Extract columns
    y = O_all[:, 2]
    x1 = O_all[:, 0]
    x2 = O_all[:, 1]
    # Clamp
    y = np.clip(y, 0, height-1)
    x1 = np.clip(x1, 0, width-1)
    x2 = np.clip(x2, 0, width-1)

    # Gather from P, Q
    p2 = P[y, x2]
    p1 = P[y, x1]
    q2 = Q[y, x2]
    q1 = Q[y, x1]

    d_line = p2 - p1
    x_contrib = (x2 * p2 - q2) - (x1 * p1 - q1)
    y_contrib = y * d_line

    # Accumulate per region
    d = np.bincount(rid_all, weights=d_line, minlength=n_regions)
    x_sum = np.bincount(rid_all, weights=x_contrib, minlength=n_regions)
    y_sum = np.bincount(rid_all, weights=y_contrib, minlength=n_regions)

    centroids = np.zeros((n_regions, 2), dtype=np.float32)
    nz = d != 0
    centroids[nz, 0] = x_sum[nz] / d[nz]
    centroids[nz, 1] = y_sum[nz] / d[nz]
    # For zero-mass regions, leave (0,0)
    return centroids


# ----------------------------
# Regular variants
# ----------------------------
def rasterize(V):
    """
    Polygon rasterization (scanlines).

    Given an ordered set of vertices V describing a polygon,
    return all the (integer) points inside the polygon.
    See http://alienryderflex.com/polygon_fill/

    Parameters:
    -----------

    V : (n,2) shaped numpy array
        Polygon vertices
    """

    n = len(V)
    X, Y = V[:, 0], V[:, 1]
    ymin = int(np.ceil(Y.min()))
    ymax = int(np.floor(Y.max()))
    #ymin = int(np.round(Y.min()))
    #ymax = int(np.round(Y.max()))
    P = []
    for y in range(ymin, ymax+1):
        segments = []
        for i in range(n):
            index1, index2 = (i-1) % n, i
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y-y1) * (x2-x1) / (y2-y1) + x1)

        segments.sort()
        for i in range(0, (2*(len(segments)//2)), 2):
            x1 = int(np.ceil(segments[i]))
            x2 = int(np.floor(segments[i+1]))
            # x1 = int(np.round(segments[i]))
            # x2 = int(np.round(segments[i+1]))
            P.extend([[x, y] for x in range(x1, x2+1)])
    if not len(P):
        return V
    return np.array(P)


def rasterize_outline(V):
    """
    Polygon outline rasterization (scanlines).

    Given an ordered set of vertices V describing a polygon,
    return all the (integer) points for the polygon outline.
    See http://alienryderflex.com/polygon_fill/

    Parameters:
    -----------

    V : (n,2) shaped numpy array
        Polygon vertices
    """
    n = len(V)
    X, Y = V[:, 0], V[:, 1]
    ymin = int(np.ceil(Y.min()))
    ymax = int(np.floor(Y.max()))
    points = np.zeros((2+(ymax-ymin)*2, 3), dtype=int)
    index = 0
    for y in range(ymin, ymax+1):
        segments = []
        for i in range(n):
            index1, index2 = (i-1) % n , i
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y-y1) * (x2-x1) / (y2-y1) + x1)
        segments.sort()
        for i in range(0, (2*(len(segments)//2)), 2):
            x1 = int(np.ceil(segments[i]))
            x2 = int(np.ceil(segments[i+1]))
            points[index] = x1, x2, y
            index += 1
    return points[:index]


def weighted_centroid_outline(V, P, Q):
    """
    Given an ordered set of vertices V describing a polygon,
    return the surface weighted centroid according to density P & Q.

    P & Q are computed relatively to density:
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    This works by first rasterizing the polygon and then
    finding the center of mass over all the rasterized points.
    """

    O = rasterize_outline(V)
    X1, X2, Y = O[:,0], O[:,1], O[:,2]

    Y = np.minimum(Y, P.shape[0]-1)
    X1 = np.minimum(X1, P.shape[1]-1)
    X2 = np.minimum(X2, P.shape[1]-1)
        
    d = (P[Y,X2]-P[Y,X1]).sum()
    x = ((X2*P[Y,X2] - Q[Y,X2]) - (X1*P[Y,X1] - Q[Y,X1])).sum()
    y = (Y * (P[Y,X2] - P[Y,X1])).sum()
    if d:
        return [x/d, y/d]
    return [x, y]
    


def uniform_centroid(V):
    """
    Given an ordered set of vertices V describing a polygon,
    returns the uniform surface centroid.

    See http://paulbourke.net/geometry/polygonmesh/
    """
    A = 0
    Cx = 0
    Cy = 0
    for i in range(len(V)-1):
        s = (V[i, 0]*V[i+1, 1] - V[i+1, 0]*V[i, 1])
        A += s
        Cx += (V[i, 0] + V[i+1, 0]) * s
        Cy += (V[i, 1] + V[i+1, 1]) * s
    Cx /= 3*A
    Cy /= 3*A
    return [Cx, Cy]


def weighted_centroid(V, D):
    """
    Given an ordered set of vertices V describing a polygon,
    return the surface weighted centroid according to density D.

    This works by first rasterizing the polygon and then
    finding the center of mass over all the rasterized points.
    """

    P = rasterize(V)
    Pi = P.astype(int)
    Pi[:, 0] = np.minimum(Pi[:, 0], D.shape[1]-1)
    Pi[:, 1] = np.minimum(Pi[:, 1], D.shape[0]-1)
    D = D[Pi[:, 1], Pi[:, 0]].reshape(len(Pi), 1)
    return ((P*D)).sum(axis=0) / D.sum()




# http://stackoverflow.com/questions/28665491/...
#    ...getting-a-bounded-polygon-coordinates-from-voronoi-cells
def in_box(points, bbox):
    return np.logical_and(
        np.logical_and(bbox[0] <= points[:, 0], points[:, 0] <= bbox[1]),
        np.logical_and(bbox[2] <= points[:, 1], points[:, 1] <= bbox[3]))


def voronoi(points, bbox):
    # See http://stackoverflow.com/questions/28665491/...
    #   ...getting-a-bounded-polygon-coordinates-from-voronoi-cells
    # See also https://gist.github.com/pv/8036995
    
    # Select points inside the bounding box
    i = in_box(points, bbox)

    # Mirror points
    points_center = points[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bbox[0] - (points_left[:, 0] - bbox[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bbox[1] + (bbox[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bbox[2] - (points_down[:, 1] - bbox[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bbox[3] + (bbox[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left, points_right, axis=0),
                                 np.append(points_down, points_up, axis=0),
                                 axis=0), axis=0)
    # Compute Voronoi
    vor = scipy.spatial.Voronoi(points)

    # Filter regions
    epsilon = 0.1
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bbox[0]-epsilon <= x <= bbox[1]+epsilon and
                       bbox[2]-epsilon <= y <= bbox[3]+epsilon):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor


def centroids(points, density, bbox, density_P=None, density_Q=None, accelerator: str = 'none'):
    """
    Given a set of point and a density array, return the set of weighted
    centroids.
    """

    # X, Y = points[:,0], points[:, 1]
    # You must ensure:
    #   0 < X.min() < X.max() < density.shape[0]
    #   0 < Y.min() < Y.max() < density.shape[1]
    
    vor = voronoi(points, bbox)
    regions = vor.filtered_regions
    
    # Build vertices list once
    vertices_list = [vor.vertices[region + [region[0]], :] for region in regions]

    # GPU batch processing (most efficient for many regions)
    if accelerator == 'cuda' and CUDA_AVAILABLE:
        P = np.ascontiguousarray(density_P)
        Q = np.ascontiguousarray(density_Q)
        centroids_array = _weighted_centroids_batch_gpu(vertices_list, P, Q, device='cuda')
        return regions, centroids_array

    # CPU batched processing (Numba reduction)
    if accelerator == 'numba' and NUMBA_AVAILABLE:
        P = np.ascontiguousarray(density_P)
        Q = np.ascontiguousarray(density_Q)
        centroids_array = _weighted_centroids_batch_numba(vertices_list, P, Q)
        return regions, centroids_array
    
    # CPU batched processing (NumPy)
    elif accelerator == 'numpy':
        P = np.ascontiguousarray(density_P)
        Q = np.ascontiguousarray(density_Q)
        centroids_array = _weighted_centroids_batch_numpy(vertices_list, P, Q)
        return regions, centroids_array

    # Default: non-batched processing
    else:
        centroids = []
        for region in regions:
            vertices = vor.vertices[region + [region[0]], :]
            # vertices = vor.filtered_points[region + [region[0]], :]

            # Full version from all the points
            # centroid = weighted_centroid(vertices, density)

            # Optimized version from only the outline
            centroid = weighted_centroid_outline(vertices, density_P, density_Q)

            centroids.append(centroid)
        return regions, np.array(centroids)
