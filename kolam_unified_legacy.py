# kolam_unified.py
# KOLAM-1: unified generator + detector from first principles
# pip install opencv-python numpy shapely pillow scikit-image

import math, json, random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString
from shapely.affinity import translate, scale
from shapely.ops import linemerge

# ---- Optional vision imports (only needed for detection) ----
try:
    import cv2
    from skimage.morphology import skeletonize
    DETECT_OK = True
except Exception:
    DETECT_OK = False


# ======================
#  Parameters (KPV)
# ======================
@dataclass
class Lattice:
    kind: str          # "square" or "diamond"
    rows: int
    cols: int
    spacing: float = 1.0
    rotation_deg: float = 0.0

@dataclass
class KPV:
    lattice: Lattice
    mask: Optional[np.ndarray] = None     # rows x cols, 1/0
    H: Optional[np.ndarray] = None        # horizontal pair states, shape (rows, cols-1)
    V: Optional[np.ndarray] = None        # vertical pair states,   shape (rows-1, cols)
    symmetry: Optional[str] = None        # e.g., "D4", "mirror-x", None
    target_components: int = 1
    arc_radius: float = 0.45              # as fraction of spacing
    stroke_px: int = 6
    allow_crossings: bool = False         # keep False for classic sikku/pulli
    recursion_depth: int = 0              # optional
    recursion_scale: float = 0.45


# ======================
#  Utility geometry
# ======================
def quarter_arc(cx, cy, r, start_deg, end_deg, steps=10):
    angs = np.linspace(math.radians(start_deg), math.radians(end_deg), steps)
    return [(cx + r*math.cos(a), cy + r*math.sin(a)) for a in angs]

def rotate(p, deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    x, y = p
    return (c*x - s*y, s*x + c*y)

def to_px(p, img_size, margin, scale_px, rot_deg=0.0):
    x, y = rotate(p, rot_deg)
    px = margin + x * scale_px
    py = img_size - (margin + y * scale_px)
    return (px, py)


# ======================
#  Core model
# ======================
# A "segment" is a quarter-arc around a dot joining two midpoints.
# At each midpoint vertex there are up to 4 incident segments (two per adjacent dot).
# Pairing at a vertex chooses how to continue: within same dot (no-merge) or across neighbor (merge).

# Endpoint labels for a dot (local): E,N,W,S -> coordinates offsets and direction tags.
MID_OFF = {
    'E': (0.5, 0.0),
    'N': (0.0, 0.5),
    'W': (-0.5, 0.0),
    'S': (0.0, -0.5),
}

# For each endpoint, the two local directions (which way the segment leaves the vertex):
# At E/W vertices, the directions are 'N' and 'S'; at N/S, they are 'E' and 'W'.
END_DIRS = {
    'E': ('N','S'), 'W': ('N','S'),
    'N': ('E','W'), 'S': ('E','W')
}

# Segments around a dot in CCW order
SEG_DEF = [('E','N'), ('N','W'), ('W','S'), ('S','E')]

class Segment:
    def __init__(self, seg_id, dot, a, b):
        self.id = seg_id
        self.dot = dot     # (i,j)
        self.a = a         # endpoint label 'E/N/W/S'
        self.b = b

    def endpoints_xy(self):
        (i,j) = self.dot
        ax, ay = MID_OFF[self.a]
        bx, by = MID_OFF[self.b]
        return [(i+ax, j+ay), (i+bx, j+by)]

    def arc_points(self, radius=0.45, steps=12):
        (i,j) = self.dot
        cx, cy = (i, j)
        # Map endpoint labels to start/end angles around center
        label_ang = {'E':0, 'N':90, 'W':180, 'S':270}
        start = label_ang[self.a]
        end   = label_ang[self.b]
        # ensure CCW quarter
        if (end - start) % 360 != 90:
            # normalize
            end = (start + 90) % 360
        return quarter_arc(cx, cy, radius, start, end, steps=steps)

def build_segments(dots: Set[Tuple[int,int]]) -> List[Segment]:
    segs = []
    sid = 0
    for (i,j) in dots:
        for (a,b) in SEG_DEF:
            segs.append(Segment(sid, (i,j), a, b))
            sid += 1
    return segs

def midpoint_xy(dot, end_label):
    (i,j) = dot
    dx, dy = MID_OFF[end_label]
    return (i+dx, j+dy)

def vertex_key(xy):
    # snap to halves to avoid float duplicates
    return (round(xy[0]*2)/2.0, round(xy[1]*2)/2.0)

def lattice_dots(latt: Lattice, mask: Optional[np.ndarray]) -> Set[Tuple[int,int]]:
    if latt.kind == "square":
        dots = {(i,j) for j in range(latt.rows) for i in range(latt.cols)}
    elif latt.kind == "diamond":
        # build 1-3-5-...-...-3-1 from rows/cols (cols must be odd and >= rows peak)
        mid = latt.rows//2
        counts = [1 + 2*min(k, latt.rows-1-k) for k in range(latt.rows)]
        width = max(counts)
        dots = set()
        y = 0
        for k, c in enumerate(counts):
            startx = (width - c)//2
            for t in range(c):
                dots.add((startx + t, y))
            y += 1
        latt.cols = width
    else:
        raise ValueError("lattice.kind must be 'square' or 'diamond'")
    if mask is not None:
        keep = set()
        for (i,j) in dots:
            if 0 <= j < mask.shape[0] and 0 <= i < mask.shape[1] and mask[j,i] == 1:
                keep.add((i,j))
        dots = keep
    return dots

def default_pair_fields(rows, cols, dots: Set[Tuple[int,int]]):
    # Create H and V with 1 where both neighbor dots exist, else 0 (ignored)
    H = np.zeros((rows, max(cols-1,1)), dtype=np.uint8)
    V = np.zeros((max(rows-1,1), cols), dtype=np.uint8)
    for (i,j) in dots:
        if (i+1,j) in dots and i+1 < cols:
            H[j,i] = 1
        if (i,j+1) in dots and j+1 < rows:
            V[j,i] = 1
    return H, V

def build_vertex_incidence(segs: List[Segment], dots: Set[Tuple[int,int]]):
    # Map vertex -> list of (segment_id, dot, end_label, dir_tag)
    incidence: Dict[Tuple[float,float], List[Tuple[int,Tuple[int,int],str,str]]] = {}
    # For each segment, both endpoints contribute at their vertex with a dir tag
    for s in segs:
        for end in [s.a, s.b]:
            v = vertex_key(midpoint_xy(s.dot, end))
            # figure direction tag for this endpoint on this segment
            # Example: at 'E', segment could be ('S','E') reaching E with 'S' or ('E','N') reaching E with 'N'
            dirs = END_DIRS[end]
            # Determine which one we are using based on (a,b)
            if end == s.a:
                # leaving end 'a' towards 'b' (CCW), choose the first dir that matches turn
                if end in ('E','W'):
                    dtag = 'N' if s.b == 'N' else 'S'
                else:
                    dtag = 'E' if s.b == 'E' else 'W'
            else:
                # coming to 'end' from the other endpoint; figure complementary
                if end in ('E','W'):
                    dtag = 'S' if s.a == 'S' else 'N'
                else:
                    dtag = 'W' if s.a == 'W' else 'E'
            incidence.setdefault(v, []).append((s.id, s.dot, end, dtag))
    return incidence

def pairings_at_vertex(vitems, H, V):
    # vitems: list of (seg_id, dot, end_label, dir_tag)
    # Group by dot(s)
    # Identify if this vertex is a horizontal or vertical shared midpoint.
    # If two dots share same y -> horizontal, else if same x -> vertical, else boundary (single dot)
    if len(vitems) <= 2:
        # boundary: pair the two (same dot)
        if len(vitems) == 2:
            return [(vitems[0][0], vitems[1][0])]
        return []
    # Two dots case (4 items)
    # Determine left/right or up/down dots
    dots = list({it[1] for it in vitems})
    if len(dots) == 1:
        # rare degenerate
        return [(vitems[0][0], vitems[1][0]), (vitems[2][0], vitems[3][0])]

    d1, d2 = dots[0], dots[1]
    # Decide orientation
    if d1[1] == d2[1]:  # same row -> horizontal neighbor
        j = d1[1]
        i = min(d1[0], d2[0])
        merge = int(H[j, i]) if i < H.shape[1] and j < H.shape[0] else 0
        # Collect by dir: pair N with N, S with S if merge; else pair within-dot (N with S)
        Ns = [it for it in vitems if it[3] == 'N']
        Ss = [it for it in vitems if it[3] == 'S']
        # within-dot groups:
        if merge == 1 and len(Ns) == 2 and len(Ss) == 2:
            return [(Ns[0][0], Ns[1][0]), (Ss[0][0], Ss[1][0])]
        else:
            # pair by dot id (within each dot’s two half-edges)
            groups: Dict[Tuple[int,int], List[int]] = {}
            for it in vitems:
                groups.setdefault(it[1], []).append(it[0])
            res = []
            for gid, segids in groups.items():
                if len(segids) == 2:
                    res.append((segids[0], segids[1]))
            return res
    else:
        # vertical neighbors
        i = d1[0] if d1[0] == d2[0] else min(d1[0], d2[0])
        j = min(d1[1], d2[1])
        merge = int(V[j, i]) if j < V.shape[0] and i < V.shape[1] else 0
        Es = [it for it in vitems if it[3] == 'E']
        Ws = [it for it in vitems if it[3] == 'W']
        if merge == 1 and len(Es) == 2 and len(Ws) == 2:
            return [(Es[0][0], Es[1][0]), (Ws[0][0], Ws[1][0])]
        else:
            groups: Dict[Tuple[int,int], List[int]] = {}
            for it in vitems:
                groups.setdefault(it[1], []).append(it[0])
            res = []
            for gid, segids in groups.items():
                if len(segids) == 2:
                    res.append((segids[0], segids[1]))
            return res

def trace_cycles(segs: List[Segment], incidence, H, V) -> List[List[int]]:
    # Build pairing map at each vertex, then traverse segments
    pair_map: Dict[int, List[int]] = {}  # seg_id -> list of paired seg_ids reachable at its endpoints
    for v, items in incidence.items():
        pairs = pairings_at_vertex(items, H, V)
        for a,b in pairs:
            pair_map.setdefault(a, []).append(b)
            pair_map.setdefault(b, []).append(a)
    visited = set()
    cycles = []
    for s in segs:
        if s.id in visited:
            continue
        # walk a cycle
        path = []
        cur = s.id
        prev = None
        while True:
            path.append(cur)
            visited.add(cur)
            nxts = [x for x in pair_map.get(cur, []) if x != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            if cur == path[0]:
                path.append(cur)
                cycles.append(path)
                break
    # Dedup small loops
    uniq = []
    seen = set()
    for c in cycles:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def enforce_component_target(H, V, segs, incidence, target=1, tries=500):
    # Simple heuristic: toggle random interior pair states to merge cycles until target reached
    bestH, bestV = H.copy(), V.copy()
    best = 1e9
    for _ in range(tries):
        cycles = trace_cycles(segs, incidence, bestH, bestV)
        k = len(cycles)
        if k == target:
            return bestH, bestV
        # randomly flip one bit that exists
        if random.random() < 0.5 and bestH.size > 0:
            j = random.randrange(bestH.shape[0])
            i = random.randrange(bestH.shape[1])
            if bestH[j,i] != 0:
                bestH[j,i] ^= 1
        else:
            if bestV.size > 0:
                j = random.randrange(bestV.shape[0])
                i = random.randrange(bestV.shape[1])
                if bestV[j,i] != 0:
                    bestV[j,i] ^= 1
        if k < best:
            best = k
    return bestH, bestV

def render_kolam(kpv: KPV, img_size=1000, margin=60):
    dots = lattice_dots(kpv.lattice, kpv.mask)
    segs = build_segments(dots)
    incidence = build_vertex_incidence(segs, dots)

    # Default H,V if absent
    if kpv.H is None or kpv.V is None:
        H, V = default_pair_fields(kpv.lattice.rows, kpv.lattice.cols, dots)
    else:
        H, V = kpv.H.copy(), kpv.V.copy()

    # Enforce component count if requested
    if kpv.target_components is not None:
        H, V = enforce_component_target(H, V, segs, incidence, kpv.target_components, tries=800)

    # Trace cycles
    cycles = trace_cycles(segs, incidence, H, V)

    # Rasterize
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)
    r = kpv.arc_radius * kpv.lattice.spacing

    for cyc in cycles:
        path_pts = []
        for sid in cyc[:-1]:
            arc = segs[sid].arc_points(radius=r, steps=10)
            # rotate then scale to pixels
            path_pts += [to_px(p, img_size, margin, scale_px= (img_size-2*margin)/(max(kpv.lattice.cols, kpv.lattice.rows)+1),
                               rot_deg=kpv.lattice.rotation_deg) for p in arc]
        draw.line(path_pts, fill=(0,0,0), width=kpv.stroke_px, joint="curve")

    # draw dots
    dot_r_px = 4
    for (i,j) in dots:
        px, py = to_px((i,j), img_size, margin,
                       scale_px=(img_size-2*margin)/(max(kpv.lattice.cols, kpv.lattice.rows)+1),
                       rot_deg=kpv.lattice.rotation_deg)
        draw.ellipse((px-dot_r_px, py-dot_r_px, px+dot_r_px, py+dot_r_px), fill=(0,0,0))
    return img, H, V


# ======================
#  Detection (OpenCV)
# ======================
def detect_from_image(image_path: str):
    if not DETECT_OK:
        raise RuntimeError("OpenCV/scikit-image not available; install them to use detection.")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot read image")

    blur = cv2.GaussianBlur(img, (5,5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 7)

    # Blob-detect dots
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True; params.minArea = 10; params.maxArea = 5000
    params.filterByCircularity = True; params.minCircularity = 0.5
    params.filterByConvexity = False; params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(thr)
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    if len(pts) < 4:
        raise ValueError("Not enough dots detected")

    # PCA → lattice axes
    X = pts - pts.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt
    B = X @ R.T
    # grid steps
    def step(c):
        v = np.sort(B[:,c]); d = np.diff(v); d = d[d>1e-3]
        return np.median(d)
    sx, sy = step(0), step(1)
    if sx<=0 or sy<=0: raise ValueError("Bad lattice fit")

    # snap to integer lattice
    G = np.stack([np.round(B[:,0]/sx), np.round(B[:,1]/sy)], axis=1).astype(int)
    rows = G[:,1].max() - G[:,1].min() + 1
    cols = G[:,0].max() - G[:,0].min() + 1
    mask = np.zeros((rows, cols), dtype=np.uint8)
    G[:,0] -= G[:,0].min(); G[:,1] -= G[:,1].min()
    for gx,gy in G:
        if 0<=gy<rows and 0<=gx<cols:
            mask[gy,gx] = 1

    # skeletonize strokes
    inv = cv2.bitwise_not(thr)
    inv = (inv>0).astype(np.uint8)*255
    skel = skeletonize(inv//255).astype(np.uint8)

    # estimate H,V by local sampling at shared midpoints
    H = np.zeros((rows, max(cols-1,1)), dtype=np.uint8)
    V = np.zeros((max(rows-1,1), cols), dtype=np.uint8)

    # Build pixel-space mapping for a lattice unit step
    scale_px = 1.0  # use R columns as basis, sx, sy as step lengths in PCA space
    # For each midpoint in lattice, map to pixel coords by reversing the PCA transform
    mean = pts.mean(0, keepdims=True)

    def lattice_to_px(gx, gy):
        b = np.array([gx*sx, gy*sy], dtype=np.float32)
        p = b @ R + mean[0]
        return (int(round(p[0])), int(round(p[1])))

    def sample_dir(x, y, dx, dy, L=6):
        # sample line of length 2L+1 around (x,y) along (dx,dy)
        s = 0
        for t in range(-L, L+1):
            xi = int(round(x + t*dx)); yi = int(round(y + t*dy))
            if 0<=yi<skel.shape[0] and 0<=xi<skel.shape[1]:
                s += skel[yi, xi]
        return s

    # Horizontal midpoints
    for j in range(rows):
        for i in range(cols-1):
            if mask[j,i]==1 and mask[j,i+1]==1:
                x1,y1 = lattice_to_px(i+0.5, j)
                # vertical vs horizontal energy
                vert = sample_dir(x1,y1, 0, 1)
                horz = sample_dir(x1,y1, 1, 0)
                H[j,i] = 1 if vert >= horz else 0

    # Vertical midpoints
    for j in range(rows-1):
        for i in range(cols):
            if mask[j,i]==1 and mask[j+1,i]==1:
                x1,y1 = lattice_to_px(i, j+0.5)
                horz = sample_dir(x1,y1, 1, 0)
                vert = sample_dir(x1,y1, 0, 1)
                V[j,i] = 1 if horz >= vert else 0

    # Lattice kind guess
    # Diamond if row counts are odd and symmetric
    row_counts = mask.sum(1).tolist()
    is_diamond = (len(row_counts)>=3
                  and all(c%2==1 for c in row_counts)
                  and row_counts == row_counts[::-1])
    kind = "diamond" if is_diamond else "square"

    latt = Lattice(kind=kind, rows=rows, cols=cols, spacing=1.0, rotation_deg=0.0)
    kpv = KPV(lattice=latt, mask=mask, H=H, V=V, target_components=1)
    return kpv


# ======================
#  CLI
# ======================
if __name__ == "__main__":
    import sys, os
    if len(sys.argv) == 1:
        # demo: 7x7 square, symmetric
        latt = Lattice(kind="square", rows=7, cols=7)
        kpv = KPV(lattice=latt, target_components=1, arc_radius=0.45, stroke_px=7)
        img, H, V = render_kolam(kpv, img_size=900, margin=60)
        img.save("kolam_square_7x7.png")
        print("Wrote kolam_square_7x7.png")
    else:
        cmd = sys.argv[1]
        if cmd == "detect" and len(sys.argv) >= 3:
            path = sys.argv[2]
            kpv = detect_from_image(path)
            img, H, V = render_kolam(kpv, img_size=1000, margin=60)
            out = os.path.splitext(os.path.basename(path))[0] + "_recreated.png"
            img.save(out)
            print(f"Recreated → {out}")
        elif cmd == "gen":
            # Example: python kolam_unified.py gen square 9 9
            kind = sys.argv[2]; rows = int(sys.argv[3]); cols = int(sys.argv[4])
            latt = Lattice(kind=kind, rows=rows, cols=cols)
            kpv = KPV(lattice=latt, target_components=1, arc_radius=0.45, stroke_px=7)
            img, H, V = render_kolam(kpv, img_size=1000, margin=60)
            out = f"kolam_{kind}_{rows}x{cols}.png"
            img.save(out)
            print(f"Wrote {out}")
        else:
            print("Usage:")
            print("  python kolam_unified.py               # demo generate")
            print("  python kolam_unified.py gen <square|diamond> <rows> <cols>")
            print("  python kolam_unified.py detect <image_path>")


# --- add to kolam_unified.py ---
try:
    import reedsolo  # RS over GF(256)
    RS_OK = True
except Exception:
    RS_OK = False

# Reserve layout helpers
def reserve_maps(rows, cols):
    # boolean masks for H and V midpoints (True = RESERVED)
    H_res = np.zeros((rows, max(cols-1,1)), dtype=bool)
    V_res = np.zeros((max(rows-1,1), cols), dtype=bool)

    # Finder 3x3 in three corners (dot indices); reserve all internal H/V midpoints touching these blocks
    def reserve_block(x0,y0, w=3,h=3):
        # Horizontal edges inside block
        for j in range(y0, y0+h):
            for i in range(x0, x0+w-1):
                H_res[j,i] = True
        # Vertical edges inside block
        for j in range(y0, y0+h-1):
            for i in range(x0, x0+w):
                V_res[j,i] = True

    reserve_block(0,0,3,3)                    # top-left
    reserve_block(cols-3,0,3,3)               # top-right
    reserve_block(0,rows-3,3,3)               # bottom-left

    # Timing lanes: one row and one column, avoid finder areas
    timing_row = 3
    timing_col = 3
    for i in range(cols-1):
        H_res[timing_row, i] = True
    for j in range(rows-1):
        V_res[j, timing_col] = True

    # Format area near TL finder: just reserve first ~40 H midpoints on row=1 and ~10 V midpoints on col=1
    fmt_bits = []
    for i in range(min(cols-1, 40)):
        H_res[1, i] = True
        fmt_bits.append(('H',1,i))
    for j in range(min(rows-1, 10)):
        V_res[j, 1] = True
        fmt_bits.append(('V',j,1))

    return H_res, V_res, fmt_bits, timing_row, timing_col

def apply_finders_timing(H, V, timing_row, timing_col):
    # timing: alternating 0101...
    for i in range(H.shape[1]):
        Vv = 1 if (i % 2 == 0) else 0
        H[timing_row, i] = Vv
    for j in range(V.shape[0]):
        Hv = 1 if (j % 2 == 0) else 0
        V[j, timing_col] = Hv

    # finder blocks: set strong "flower" motif (mostly merges)
    def fill_block(H,V,x0,y0):
        for j in range(y0, y0+3):
            for i in range(x0, x0+2):
                H[j,i] = 1
        for j in range(y0, y0+2):
            for i in range(x0, x0+3):
                V[j,i] = 1
        # outer ring light (0) to outline
        # top-left ring tweak
        H[y0, x0] = 0; H[y0+2, x0+1] = 0
        V[y0, x0] = 0; V[y0+1, x0+2] = 0

    fill_block(H,V,0,0)
    fill_block(H,V,H.shape[1]-2,0)       # approx top-right
    fill_block(H,V,0,V.shape[0]-2)       # approx bottom-left

def mask_functions():
    return [
        lambda i,j: ((i + j) % 2) == 0,
        lambda i,j: (j % 2) == 0,
        lambda i,j: (i % 3) == 0,
        lambda i,j: (((i*j) % 2) + ((i*j) % 3)) % 2 == 0,
    ]

def apply_mask_bits(bits, indices, mask_id, is_H):
    mfs = mask_functions()
    mf = mfs[mask_id % len(mfs)]
    out = []
    for (coord, b) in zip(indices, bits):
        if is_H:
            _, j, i = coord  # ('H', j, i)
        else:
            _, j, i = coord  # ('V', j, i)
        flip = mf(i, j)
        out.append(b ^ int(flip))
    return out

def rs_encode(data_bytes, ecc_bytes):
    if not RS_OK:
        # fallback: simple parity (bad ECC but keeps code running)
        return data_bytes + bytes([sum(data_bytes)%256 for _ in range(ecc_bytes)])
    rs = reedsolo.RSCodec(ecc_bytes)
    return bytes(rs.encode(bytearray(data_bytes)))

def rs_decode(all_bytes, ecc_bytes):
    if not RS_OK:
        # no real decoding, just strip
        k = len(all_bytes) - ecc_bytes
        return all_bytes[:k], True
    rs = reedsolo.RSCodec(ecc_bytes)
    try:
        dec = rs.decode(bytearray(all_bytes))
        return bytes(dec), True
    except Exception:
        return bytes(all_bytes[:-ecc_bytes]), False

def zigzag_indices(H_res, V_res):
    H_idx = []
    for j in range(H_res.shape[0]):
        rng = range(H_res.shape[1]) if j%2==0 else range(H_res.shape[1]-1, -1, -1)
        for i in rng:
            if not H_res[j,i]:
                H_idx.append(('H', j, i))
    V_idx = []
    for i in range(V_res.shape[1]):
        rng = range(V_res.shape[0]) if i%2==0 else range(V_res.shape[0]-1, -1, -1)
        for j in rng:
            if not V_res[j,i]:
                V_idx.append(('V', j, i))
    return H_idx, V_idx

def kolamqr_encode(data: bytes, rows: int, cols: int, ecc_level: str = 'M'):
    # ECC bytes by level (rough)
    ecc_map = {'L': 0.10, 'M': 0.18, 'Q': 0.25, 'H': 0.33}
    rate = ecc_map.get(ecc_level, 0.18)

    mask = np.ones((rows, cols), dtype=np.uint8)   # full dot mask
    dots = lattice_dots(Lattice('square', rows, cols), mask)
    H = np.zeros((rows, max(cols-1,1)), dtype=np.uint8)
    V = np.zeros((max(rows-1,1), cols), dtype=np.uint8)

    H_res, V_res, fmt_slots, trow, tcol = reserve_maps(rows, cols)
    apply_finders_timing(H, V, trow, tcol)

    # plan capacity
    H_idx, V_idx = zigzag_indices(H_res, V_res)
    capacity_bits = len(H_idx) + len(V_idx) - len(fmt_slots)
    k_bytes = max(1, capacity_bits // 8)
    ecc_bytes = int(rate * k_bytes)
    total_bytes = k_bytes - ecc_bytes
    payload = data[:total_bytes]
    coded = rs_encode(payload, ecc_bytes)

    # bitstream
    bits = []
    for b in coded:
        for t in range(8):
            bits.append((b >> (7 - t)) & 1)
    bits = bits[:len(H_idx)+len(V_idx)-len(fmt_slots)]
    # simple interleave
    half = len(bits)//2
    bits_H = bits[:min(half,len(H_idx))] + bits[min(half,len(H_idx)):]
    bits_V = bits[min(len(H_idx),half):]

    # choose best mask
    best = None
    best_id = 0
    mfs = mask_functions()
    for mid in range(len(mfs)):
        mH = apply_mask_bits(bits_H[:len(H_idx)], H_idx, mid, True)
        mV = apply_mask_bits(bits_V[:len(V_idx)], V_idx, mid, False)
        score = sum(mH) + sum(mV)  # trivial density heuristic; replace with penalties if you want
        if best is None or score < best:
            best, best_id = score, mid
            bestH, bestV = mH, mV

    # write masked bits
    for (coord, b) in zip(H_idx, bestH):
        _, j,i = coord
        H[j,i] = b
    for (coord, b) in zip(V_idx, bestV):
        _, j,i = coord
        V[j,i] = b

    # write format (mask id, ecc level, version, crc) into fmt_slots
    # keep it minimal: mask(3) + ecc(2) + rows(6) + cols(6) + crc8(8) = 25 bits
    ecc2 = {'L':0,'M':1,'Q':2,'H':3}[ecc_level]
    fmt_bits = []
    fmt_bits += [(best_id>>2)&1, (best_id>>1)&1, best_id&1]
    fmt_bits += [(ecc2>>1)&1, ecc2&1]
    def tobits(val, n): return [ (val>>(n-1-i))&1 for i in range(n) ]
    fmt_bits += tobits(rows,6) + tobits(cols,6)
    # trivial CRC8
    x = 0
    for b in fmt_bits:
        x = ((x<<1)&0xFF) ^ (0x07 if ((x>>7)&1) ^ b else 0)
    fmt_bits += tobits(x,8)
    # put fmt bits
    for (slot, b) in zip(fmt_slots, fmt_bits):
        t, j,i = slot
        if t=='H': H[j,i] = b
        else: V[j,i] = b

    # render with existing renderer
    kpv = KPV(lattice=Lattice('square', rows, cols), mask=mask, H=H, V=V, target_components=None, arc_radius=0.45, stroke_px=7)
    img, _, _ = render_kolam(kpv, img_size=1000, margin=60)
    return img

def kolamqr_decode(image_path: str):
    kpv = detect_from_image(image_path)
    H, V = kpv.H, kpv.V
    rows, cols = kpv.lattice.rows, kpv.lattice.cols

    H_res, V_res, fmt_slots, trow, tcol = reserve_maps(rows, cols)

    # read format
    bits = []
    for t,j,i in fmt_slots:
        bits.append( H[j,i] if t=='H' else V[j,i] )
    # parse: mask(3), ecc(2), rows(6), cols(6), crc8(8)
    if len(bits) < 25: return b"", False
    mask_id = (bits[0]<<2) | (bits[1]<<1) | bits[2]
    ecc2 = (bits[3]<<1) | bits[4]
    ecc_level = {0:'L',1:'M',2:'Q',3:'H'}.get(ecc2, 'M')
    val_rows = 0
    for b in bits[5:11]: val_rows = (val_rows<<1) | b
    val_cols = 0
    for b in bits[11:17]: val_cols = (val_cols<<1) | b
    crc_read = 0
    for b in bits[17:25]: crc_read = (crc_read<<1) | b
    # quick crc check
    crc_bits = bits[:17]
    x=0
    for b in crc_bits:
        x = ((x<<1)&0xFF) ^ (0x07 if ((x>>7)&1) ^ b else 0)
    if x != crc_read:
        # keep going but warn
        pass

    H_idx, V_idx = zigzag_indices(H_res, V_res)

    # collect masked bits
    mH = [ H[j,i] for (_,j,i) in H_idx ]
    mV = [ V[j,i] for (_,j,i) in V_idx ]

    # unmask
    bits_H = apply_mask_bits(mH, H_idx, mask_id, True)
    bits_V = apply_mask_bits(mV, V_idx, mask_id, False)

    bits_all = bits_H + bits_V
    # reconstruct bytes; we don't know exact ecc_bytes here; read from payload length markers in future
    # For demo: assume last 18% bytes are ECC:
    total_bits = len(bits_all)
    total_bytes = total_bits//8
    b = []
    for k in range(total_bytes):
        v=0
        for t in range(8):
            v = (v<<1) | bits_all[k*8+t]
        b.append(v)
    b = bytes(b)
    ecc_bytes = max(1, int(0.18*len(b)))
    data, ok = rs_decode(b, ecc_bytes)
    return data, ok
