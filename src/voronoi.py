# -----------------------------------------------------------------------------
# Weighted Voronoi Stippler
# Copyright (2017) Nicolas P. Rougier - BSD license
# -----------------------------------------------------------------------------
import numpy as np
import scipy.spatial

# Optional acceleration with Numba (CPU JIT)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


# ----------------------------
# Numba-accelerated variants
# ----------------------------
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

    @njit(cache=True)
    def _weighted_centroid_outline_numba(V, P, Q):
        O = _rasterize_outline_numba(V)
        if O.shape[0] == 0:
            return np.array([0.0, 0.0])

        height = P.shape[0]
        width = P.shape[1]

        d = 0.0
        x_sum = 0.0
        y_sum = 0.0
        for k in range(O.shape[0]):
            y = O[k, 2]
            if y < 0:
                continue
            if y >= height:
                y = height - 1
            x1 = O[k, 0]
            x2 = O[k, 1]
            if x1 >= width:
                x1 = width - 1
            if x2 >= width:
                x2 = width - 1
            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0

            p2 = P[y, x2]
            p1 = P[y, x1]
            d_line = p2 - p1
            d += d_line
            x_sum += (x2 * p2 - Q[y, x2]) - (x1 * p1 - Q[y, x1])
            y_sum += y * d_line

        if d != 0.0:
            return np.array([x_sum / d, y_sum / d])
        return np.array([x_sum, y_sum])


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
    centroids = []
    for region in regions:
        vertices = vor.vertices[region + [region[0]], :]
        # vertices = vor.filtered_points[region + [region[0]], :]

        # Full version from all the points
        # centroid = weighted_centroid(vertices, density)

        # Optimized version from only the outline
        if accelerator == 'numba' and NUMBA_AVAILABLE:
            # Ensure contiguous arrays of proper dtype for numba
            V = np.ascontiguousarray(vertices, dtype=np.float64)
            P = np.ascontiguousarray(density_P)
            Q = np.ascontiguousarray(density_Q)
            centroid = _weighted_centroid_outline_numba(V, P, Q)
        else:
            centroid = weighted_centroid_outline(vertices, density_P, density_Q)

        centroids.append(centroid)
    return regions, np.array(centroids)
