# kolam_detector.py
import cv2, numpy as np
from typing import List, Tuple
from A3_gen import generate_kolam, diamond_lattice, rect_lattice

def detect_dots(image_path: str) -> List[Tuple[float,float]]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot read image")

    # binarize robustly
    blur = cv2.GaussianBlur(img, (5,5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 35, 7)

    # detect circular blobs (dots) with SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thr)

    pts = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    if len(pts) < 4:
        raise ValueError("Not enough dots detected")

    return pts

def fit_lattice(pts: List[Tuple[float,float]]):
    # Estimate axis directions by PCA, then project to an orthogonal grid
    X = np.array(pts, dtype=np.float32)
    X -= X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt  # rows are basis vectors
    B = X @ R.T
    # Estimate grid step as median nearest-neighbor spacing along each axis
    def axis_step(col):
        vals = np.sort(B[:,col])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 1e-3]
        return np.median(diffs)
    sx = axis_step(0)
    sy = axis_step(1)
    if sx <= 0 or sy <= 0:
        raise ValueError("Bad grid step")

    # Snap to integer lattice
    G = np.stack([np.round(B[:,0]/sx), np.round(B[:,1]/sy)], axis=1).astype(int)
    G_unique = np.unique(G, axis=0)

    # Classify layout: rectangular or diamond (odd rows increasing then decreasing)
    rows = {}
    for gx, gy in G_unique:
        rows.setdefault(gy, []).append(gx)
    row_counts = [len(sorted(v)) for k,v in sorted(rows.items())]
    is_diamond = (len(row_counts) >= 3 and
                  all(c%2==1 for c in row_counts) and
                  row_counts == row_counts[::-1] or
                  row_counts == sorted(row_counts) + sorted(row_counts[-2::-1]))

    if is_diamond:
        # Build diamond spec as counts per row
        counts = row_counts
        return ("diamond", counts)
    else:
        # Rectangular m x n
        n_rows = len(row_counts)
        n_cols = max(row_counts)
        return ("rect", (n_rows, n_cols))

def recreate(image_path: str, out_path="kolam_recreated.png"):
    pts = detect_dots(image_path)
    kind, spec = fit_lattice(pts)
    if kind == "diamond":
        dots = []
        # normalize to your generator's coordinate convention
        dots = []
        y = 0
        for k in spec:
            startx = -(k//2)
            for t in range(k):
                dots.append((startx + t, y))
            y += 1
        # shift to positive
        minx = min(i for i,_ in dots)
        miny = min(j for _,j in dots)
        dots = [(i-minx, j-miny) for i,j in dots]
    else:
        n_rows, n_cols = spec
        dots = [(i, j) for i in range(n_cols) for j in range(n_rows)]

    img = generate_kolam(dots, img_size=1000, scale_px=60, stroke=8)
    img.save(out_path)
    print(f"Saved {out_path} using inferred {kind} layout: {spec}")

if __name__ == "__main__":
    # Example:
    # python kolam_detector.py path/to/photo.jpg
    import sys
    if len(sys.argv) < 2:
        print("Usage: python kolam_detector.py <image>")
        sys.exit(1)
    recreate(sys.argv[1], "kolam_recreated.png")
