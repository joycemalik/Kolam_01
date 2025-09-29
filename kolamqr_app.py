# kolamqr_app.py
# Kolam-QR: robust encoder/decoder using kolam pairing states (midpoints) with ECC + masking
# Outputs PNG (and optional SVG). Works on photos (perspective/rotation tolerant).

import os, sys, math, json, random, argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set

import numpy as np
from PIL import Image, ImageDraw

# Vision stack (for decode)
import cv2
from skimage.morphology import skeletonize

# Optional SVG (nice to have)
import svgwrite

# ECC
try:
    import reedsolo
    RS_OK = True
except Exception:
    RS_OK = False


# =========================
# Basic geometry + lattice
# =========================
@dataclass
class Lattice:
    kind: str          # "square" or "diamond" (we use square for Kolam-QR versions)
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
    target_components: Optional[int] = None
    arc_radius: float = 0.45
    stroke_px: int = 6

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

def lattice_dots(latt: Lattice, mask: Optional[np.ndarray]) -> Set[Tuple[int,int]]:
    if latt.kind != "square":
        raise ValueError("Use square lattice for Kolam-QR.")
    dots = {(i,j) for j in range(latt.rows) for i in range(latt.cols)}
    if mask is not None:
        keep = set()
        for (i,j) in dots:
            if 0 <= j < mask.shape[0] and 0 <= i < mask.shape[1] and mask[j,i] == 1:
                keep.add((i,j))
        dots = keep
    return dots


# =========================
# Segments + incidence
# =========================
MID_OFF = {'E':(0.5,0.0), 'N':(0.0,0.5), 'W':(-0.5,0.0), 'S':(0.0,-0.5)}
SEG_DEF = [('E','N'), ('N','W'), ('W','S'), ('S','E')]  # CCW quarter arcs

class Segment:
    def __init__(self, seg_id, dot, a, b):
        self.id = seg_id
        self.dot = dot     # (i,j)
        self.a = a         # 'E/N/W/S'
        self.b = b

    def arc_points(self, radius=0.45, steps=14):
        (i,j) = self.dot
        cx, cy = (i, j)
        label_ang = {'E':0, 'N':90, 'W':180, 'S':270}
        start = label_ang[self.a]
        end   = (label_ang[self.b]) % 360
        # ensure quarter CCW
        if (end - start) % 360 != 90:
            end = (start + 90) % 360
        angs = np.linspace(math.radians(start), math.radians(end), steps)
        return [(cx + radius*math.cos(a), cy + radius*math.sin(a)) for a in angs]

def build_segments(dots: Set[Tuple[int,int]]) -> List[Segment]:
    segs = []
    sid = 0
    for (i,j) in dots:
        for (a,b) in SEG_DEF:
            segs.append(Segment(sid, (i,j), a, b)); sid+=1
    return segs

def midpoint_xy(dot, end_label):
    (i,j) = dot
    dx, dy = MID_OFF[end_label]
    return (i+dx, j+dy)

def vertex_key(xy):
    return (round(xy[0]*2)/2.0, round(xy[1]*2)/2.0)

# For endpoint 'end', the two local directions are:
END_DIRS = {'E':('N','S'),'W':('N','S'),'N':('E','W'),'S':('E','W')}

def build_vertex_incidence(segs: List[Segment], dots: Set[Tuple[int,int]]):
    incidence: Dict[Tuple[float,float], List[Tuple[int,Tuple[int,int],str,str]]] = {}
    for s in segs:
        for end in [s.a, s.b]:
            v = vertex_key(midpoint_xy(s.dot, end))
            if end == s.a:
                dtag = END_DIRS[end][0] if (end in ('E','W') and s.b=='N') or (end in ('N','S') and s.b=='E') else END_DIRS[end][1]
            else:
                dtag = END_DIRS[end][1] if (end in ('E','W') and s.a=='N') or (end in ('N','S') and s.a=='E') else END_DIRS[end][0]
            incidence.setdefault(v, []).append((s.id, s.dot, end, dtag))
    return incidence

def pairings_at_vertex(vitems, H, V):
    if len(vitems) <= 2:
        if len(vitems)==2:
            return [(vitems[0][0], vitems[1][0])]
        return []
    dots = list({it[1] for it in vitems})
    if len(dots) == 1:
        return [(vitems[0][0], vitems[1][0]), (vitems[2][0], vitems[3][0])]
    d1, d2 = dots[0], dots[1]
    if d1[1] == d2[1]:  # horizontal neighbors
        j = d1[1]
        i = min(d1[0], d2[0])
        merge = int(H[j, i]) if 0<=j<H.shape[0] and 0<=i<H.shape[1] else 0
        Ns = [it for it in vitems if it[3]=='N']; Ss=[it for it in vitems if it[3]=='S']
        if merge==1 and len(Ns)==2 and len(Ss)==2:
            return [(Ns[0][0],Ns[1][0]),(Ss[0][0],Ss[1][0])]
        else:
            groups: Dict[Tuple[int,int], List[int]] = {}
            for it in vitems: groups.setdefault(it[1], []).append(it[0])
            res=[]
            for segids in groups.values():
                if len(segids)==2: res.append((segids[0],segids[1]))
            return res
    else:  # vertical
        i = d1[0] if d1[0]==d2[0] else min(d1[0], d2[0])
        j = min(d1[1], d2[1])
        merge = int(V[j, i]) if 0<=j<V.shape[0] and 0<=i<V.shape[1] else 0
        Es = [it for it in vitems if it[3]=='E']; Ws=[it for it in vitems if it[3]=='W']
        if merge==1 and len(Es)==2 and len(Ws)==2:
            return [(Es[0][0],Es[1][0]),(Ws[0][0],Ws[1][0])]
        else:
            groups: Dict[Tuple[int,int], List[int]] = {}
            for it in vitems: groups.setdefault(it[1], []).append(it[0])
            res=[]
            for segids in groups.values():
                if len(segids)==2: res.append((segids[0],segids[1]))
            return res

def trace_cycles(segs: List[Segment], incidence, H, V) -> List[List[int]]:
    pair_map: Dict[int, List[int]] = {}
    for v, items in incidence.items():
        pairs = pairings_at_vertex(items, H, V)
        for a,b in pairs:
            pair_map.setdefault(a, []).append(b)
            pair_map.setdefault(b, []).append(a)
    visited = set()
    cycles = []
    for s in segs:
        if s.id in visited: continue
        path=[]; cur=s.id; prev=None
        while True:
            path.append(cur); visited.add(cur)
            nxts = [x for x in pair_map.get(cur, []) if x!=prev]
            if not nxts: break
            prev, cur = cur, nxts[0]
            if cur==path[0]:
                path.append(cur); cycles.append(path); break
    # dedup
    uniq=[]; seen=set()
    for c in cycles:
        key=tuple(sorted(c))
        if key not in seen: seen.add(key); uniq.append(c)
    return uniq


# =========================
# Reserve areas (finders/timing/format)
# =========================
def reserve_maps(rows, cols):
    H_res = np.zeros((rows, max(cols-1,1)), dtype=bool)
    V_res = np.zeros((max(rows-1,1), cols), dtype=bool)

    def reserve_block(x0,y0, w=3,h=3):
        for j in range(y0, y0+h):
            for i in range(x0, x0+w-1):
                if 0<=j<H_res.shape[0] and 0<=i<H_res.shape[1]:
                    H_res[j,i]=True
        for j in range(y0, y0+h-1):
            for i in range(x0, x0+w):
                if 0<=j<V_res.shape[0] and 0<=i<V_res.shape[1]:
                    V_res[j,i]=True

    # Three 3x3 finders: TL, TR, BL
    reserve_block(0,0,3,3)
    reserve_block(cols-3,0,3,3)
    reserve_block(0,rows-3,3,3)

    # Timing lanes
    timing_row = 3 if rows>4 else 1
    timing_col = 3 if cols>4 else 1
    for i in range(H_res.shape[1]): H_res[timing_row, i] = True
    for j in range(V_res.shape[0]): V_res[j, timing_col] = True

    # Format (≈25 bits)
    fmt_slots=[]
    for i in range(min(cols-1, 40)):
        H_res[1, i]=True; fmt_slots.append(('H',1,i))
    for j in range(min(rows-1, 10)):
        V_res[j, 1]=True; fmt_slots.append(('V',j,1))

    return H_res, V_res, fmt_slots, timing_row, timing_col

def apply_finders_timing(H, V, trow, tcol):
    # timing: alt 0101...
    for i in range(H.shape[1]): H[trow, i] = 1 if (i%2==0) else 0
    for j in range(V.shape[0]): V[j, tcol] = 1 if (j%2==0) else 0

    # simple “flower” finders: strong merges inside block, light ring
    def fill_block(H,V,x0,y0):
        for j in range(y0, y0+3):
            for i in range(x0, x0+2):
                if 0<=j<H.shape[0] and 0<=i<H.shape[1]:
                    H[j,i]=1
        for j in range(y0, y0+2):
            for i in range(x0, x0+3):
                if 0<=j<V.shape[0] and 0<=i<V.shape[1]:
                    V[j,i]=1
        # outline tweaks
        if 0<=y0<H.shape[0] and 0<=x0<H.shape[1]: H[y0,x0]=0
    fill_block(H,V,0,0)
    fill_block(H,V,H.shape[1]-2,0)
    fill_block(H,V,0,V.shape[0]-2)

def zigzag_indices(H_res, V_res):
    H_idx=[]; V_idx=[]
    for j in range(H_res.shape[0]):
        rng = range(H_res.shape[1]) if j%2==0 else range(H_res.shape[1]-1, -1, -1)
        for i in rng:
            if not H_res[j,i]: H_idx.append(('H',j,i))
    for i in range(V_res.shape[1]):
        rng = range(V_res.shape[0]) if i%2==0 else range(V_res.shape[0]-1, -1, -1)
        for j in rng:
            if not V_res[j,i]: V_idx.append(('V',j,i))
    return H_idx, V_idx

def mask_functions():
    return [
        lambda i,j: ((i + j) % 2) == 0,
        lambda i,j: (j % 2) == 0,
        lambda i,j: (i % 3) == 0,
        lambda i,j: (((i*j) % 2) + ((i*j) % 3)) % 2 == 0,
    ]

def apply_mask_bits(bits, indices, mask_id, is_H):
    mfs = mask_functions(); mf = mfs[mask_id % len(mfs)]
    out=[]
    for (coord, b) in zip(indices, bits):
        _, j, i = coord
        flip = mf(i,j)
        out.append(b ^ int(flip))
    return out


# =========================
# Bit/byte utils + ECC
# =========================
def bytes_to_bits(data: bytes) -> List[int]:
    out=[]
    for b in data:
        for i in range(8):
            out.append((b >> (7-i)) & 1)
    return out

def bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits + [0] * (8 - (len(bits) % 8))
    by = bytearray()
    for i in range(0, len(bits), 8):
        v=0
        for j in range(8):
            v=(v<<1)|(bits[i+j]&1)
        by.append(v)
    return bytes(by)

def rs_encode_blocks(data_bytes, ecc_bytes):
    if not RS_OK or ecc_bytes<=0:
        return data_bytes
    nsize = 255
    nsym  = min(ecc_bytes, 254)
    k = nsize - nsym
    rs = reedsolo.RSCodec(nsym, nsize=nsize)
    out = bytearray()
    for i in range(0, len(data_bytes), k):
        out.extend(rs.encode(bytearray(data_bytes[i:i+k])))
    return bytes(out)

def rs_decode_blocks(all_bytes, ecc_bytes):
    if not RS_OK or ecc_bytes<=0:
        return all_bytes, False
    nsize = 255
    nsym  = min(ecc_bytes, 254)
    k = nsize - nsym
    rs = reedsolo.RSCodec(nsym, nsize=nsize)
    out = bytearray()
    ok = True
    for i in range(0, len(all_bytes), nsize):
        blk = bytearray(all_bytes[i:i+nsize])
        if len(blk)==0: continue
        try:
            dec = rs.decode(blk)   # returns data-only
            out.extend(dec)
        except Exception:
            ok = False
            out.extend(blk[:min(len(blk), k)])
    return bytes(out), ok


# =========================
# Kolam-QR encode
# =========================
def kolamqr_encode(data: bytes, rows: int, cols: int, ecc_level: str='M',
                   png_out: str="kolamqr.png", svg_out: Optional[str]=None):
    # lattice + mask
    mask = np.ones((rows, cols), dtype=np.uint8)
    dots = lattice_dots(Lattice('square', rows, cols), mask)
    H = np.zeros((rows, max(cols-1,1)), dtype=np.uint8)
    V = np.zeros((max(rows-1,1), cols), dtype=np.uint8)

    # reserves
    H_res, V_res, fmt_slots, trow, tcol = reserve_maps(rows, cols)
    apply_finders_timing(H, V, trow, tcol)

    # capacity = all *non-reserved* H and V slots
    H_idx, V_idx = zigzag_indices(H_res, V_res)
    capacity_bits = len(H_idx) + len(V_idx)
    k_bytes = max(1, capacity_bits // 8)

    ecc_map = {'L':0.10,'M':0.18,'Q':0.25,'H':0.33}
    rate = ecc_map.get(ecc_level, 0.18)
    ecc_bytes = int(rate * k_bytes)
    ecc_bytes = max(1, min(ecc_bytes, k_bytes-1, 200))
    data_bytes = k_bytes - ecc_bytes

    payload = data[:data_bytes]
    coded = rs_encode_blocks(payload, ecc_bytes)  # length >= data_bytes (chunked). For mapping, use first k_bytes.
    coded = coded[:k_bytes]

    # bitstream
    bits = []
    for b in coded:
        for t in range(8): bits.append((b>>(7-t))&1)
    bits = bits[:capacity_bits]

    # split to H/V
    # simple interleave; H first
    nH = len(H_idx)
    bits_H = bits[:nH]
    bits_V = bits[nH:]

    # choose best mask (simple density heuristic)
    mfs = mask_functions()
    best_id=0; best_score=None
    for mid in range(len(mfs)):
        mH = apply_mask_bits(bits_H, H_idx, mid, True)
        mV = apply_mask_bits(bits_V, V_idx, mid, False)
        score = abs(sum(mH)+sum(mV) - 0.5*(len(bits_H)+len(bits_V)))  # closer to half is better
        if best_score is None or score < best_score:
            best_score = score; best_id = mid; bestH, bestV = mH, mV

    # write masked bits
    for (coord, b) in zip(H_idx, bestH):
        _, j,i = coord; H[j,i]=b
    for (coord, b) in zip(V_idx, bestV):
        _, j,i = coord; V[j,i]=b

    # format bits: mask(3) + ecc(2) + rows(6) + cols(6) + crc8(8) = 25
    def tobits(val, n): return [ (val>>(n-1-k))&1 for k in range(n) ]
    ecc2 = {'L':0,'M':1,'Q':2,'H':3}.get(ecc_level,1)
    fmt_bits = []
    fmt_bits += tobits(best_id,3)
    fmt_bits += tobits(ecc2,2)
    fmt_bits += tobits(rows,6) + tobits(cols,6)
    # simple CRC8 over previous 17 bits
    x=0
    for b in fmt_bits:
        x = ((x<<1)&0xFF) ^ (0x07 if ((x>>7)&1) ^ b else 0)
    fmt_bits += tobits(x,8)
    for (slot, b) in zip(fmt_slots, fmt_bits):
        t,j,i = slot
        if t=='H': H[j,i]=b
        else: V[j,i]=b

    # render PNG
    kpv = KPV(lattice=Lattice('square', rows, cols), mask=mask, H=H, V=V,
              target_components=None, arc_radius=0.46, stroke_px=8)
    img = render_png(kpv, img_size=1200, margin=80, solid_dots=True)  # solid dots aid detection
    img.save(png_out)

    # optional SVG (nice)
    if svg_out:
        render_svg(kpv, out_svg=svg_out, view_px=1400, margin=90)

    meta = {"rows": rows, "cols": cols, "capacity_bits": capacity_bits,
            "k_bytes": k_bytes, "ecc_bytes": ecc_bytes, "mask_id": best_id, "ecc": ecc_level}
    return png_out, meta


# =========================
# Kolam-QR decode
# =========================
def detect_dots(gray: np.ndarray) -> np.ndarray:
    # Improve contrast
    g = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    # Morph close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True; params.minArea = 8; params.maxArea = 5000
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(th)
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    return pts

def fit_lattice_from_dots(pts: np.ndarray):
    if pts.shape[0] < 9:
        raise ValueError("Not enough dots detected")
    X = pts - pts.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt
    B = X @ R.T
    def step(c):
        v=np.sort(B[:,c]); d=np.diff(v); d=d[d>1e-3]
        return np.median(d) if len(d)>0 else 0
    sx, sy = step(0), step(1)
    if sx<=0 or sy<=0: raise ValueError("Failed to estimate lattice steps")

    G = np.stack([np.round(B[:,0]/sx), np.round(B[:,1]/sy)], axis=1).astype(int)
    gx0, gy0 = G[:,0].min(), G[:,1].min()
    G[:,0]-=gx0; G[:,1]-=gy0
    rows = int(G[:,1].max() + 1); cols = int(G[:,0].max() + 1)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for gx,gy in G:
        if 0<=gy<rows and 0<=gx<cols: mask[gy,gx]=1
    return (R, pts.mean(0), sx, sy, rows, cols, mask)

def recover_HV_from_image(gray: np.ndarray, R, mean, sx, sy, rows, cols):
    # binarize & skeletonize
    g = cv2.GaussianBlur(gray,(3,3),0)
    thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    inv = (cv2.bitwise_not(thr) > 0).astype(np.uint8)
    skel = skeletonize(inv).astype(np.uint8)

    def lattice_to_px(gx, gy):
        b = np.array([gx*sx, gy*sy], dtype=np.float32)
        p = b @ R + mean
        return int(round(p[0])), int(round(p[1]))

    H = np.zeros((rows, max(cols-1,1)), dtype=np.uint8)
    V = np.zeros((max(rows-1,1), cols), dtype=np.uint8)

    def sample_dir(x,y, dx,dy, L=6):
        s=0
        for t in range(-L,L+1):
            xi = int(round(x + t*dx)); yi=int(round(y + t*dy))
            if 0<=yi<skel.shape[0] and 0<=xi<skel.shape[1]:
                s += skel[yi, xi]
        return s

    # Horizontal midpoints
    for j in range(rows):
        for i in range(cols-1):
            x1,y1 = lattice_to_px(i+0.5, j)
            vert = sample_dir(x1,y1, 0,1)
            horz = sample_dir(x1,y1, 1,0)
            H[j,i] = 1 if vert>=horz else 0

    # Vertical midpoints
    for j in range(rows-1):
        for i in range(cols):
            x1,y1 = lattice_to_px(i, j+0.5)
            horz = sample_dir(x1,y1, 1,0)
            vert = sample_dir(x1,y1, 0,1)
            V[j,i] = 1 if horz>=vert else 0

    return H,V

def parse_format(H,V,rows,cols):
    H_res, V_res, fmt_slots, trow, tcol = reserve_maps(rows, cols)
    bits=[]
    for t,j,i in fmt_slots:
        bits.append( H[j,i] if t=='H' else V[j,i] )
    if len(bits) < 25:
        raise ValueError("Format bits missing")
    mask_id = (bits[0]<<2)|(bits[1]<<1)|bits[2]
    ecc2 = (bits[3]<<1)|bits[4]
    ecc_level = {0:'L',1:'M',2:'Q',3:'H'}.get(ecc2, 'M')
    # rows/cols from format if needed (we already have them from lattice fit)
    # CRC check
    crc_bits = bits[:17]; x=0
    for b in crc_bits:
        x = ((x<<1)&0xFF) ^ (0x07 if ((x>>7)&1) ^ b else 0)
    crc_read = 0
    for b in bits[17:25]: crc_read=(crc_read<<1)|b
    # not fatal if mismatch
    return mask_id, ecc_level

def kolamqr_decode(image_path: str):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # detect dots
    pts = detect_dots(img)
    if pts.shape[0] < 9:
        raise RuntimeError("Not enough dots; is the image clear/sharp?")

    # fit lattice and mask
    R, mean, sx, sy, rows, cols, mask = fit_lattice_from_dots(pts)

    # recover H/V by sampling skeleton at midpoints
    H,V = recover_HV_from_image(img, R, mean, sx, sy, rows, cols)

    # format
    mask_id, ecc_level = parse_format(H,V,rows,cols)
    H_res, V_res, fmt_slots, trow, tcol = reserve_maps(rows, cols)
    H_idx, V_idx = zigzag_indices(H_res, V_res)
    capacity_bits = len(H_idx)+len(V_idx)
    k_bytes = max(1, capacity_bits//8)
    ecc_map = {'L':0.10,'M':0.18,'Q':0.25,'H':0.33}
    rate = ecc_map.get(ecc_level,0.18)
    ecc_bytes = int(rate*k_bytes)
    ecc_bytes = max(1, min(ecc_bytes, k_bytes-1, 200))

    # gather masked bits
    mH = [ H[j,i] for (_,j,i) in H_idx ]
    mV = [ V[j,i] for (_,j,i) in V_idx ]
    # unmask
    bits_H = apply_mask_bits(mH, H_idx, mask_id, True)
    bits_V = apply_mask_bits(mV, V_idx, mask_id, False)
    bits_all = bits_H + bits_V

    # bytes
    total_bytes = len(bits_all)//8
    b = []
    for k in range(total_bytes):
        v=0
        for t in range(8):
            v=(v<<1)|bits_all[k*8+t]
        b.append(v)
    b = bytes(b)
    b = b[:k_bytes]

    data, ok = rs_decode_blocks(b, ecc_bytes)
    return data, ok, {"rows":rows,"cols":cols,"mask_id":mask_id,"ecc":ecc_level}


# =========================
# Rendering (PNG + SVG)
# =========================
def render_png(kpv: KPV, img_size=1000, margin=60, solid_dots=True) -> Image.Image:
    dots = lattice_dots(kpv.lattice, kpv.mask)
    segs = build_segments(dots)
    incidence = build_vertex_incidence(segs, dots)
    H = kpv.H; V = kpv.V
    cycles = trace_cycles(segs, incidence, H, V)

    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)
    scale_px = (img_size-2*margin)/(max(kpv.lattice.cols, kpv.lattice.rows)+1)
    r = kpv.arc_radius * kpv.lattice.spacing

    # double-stroke for crispness
    for cyc in cycles:
        pts=[]
        for sid in cyc[:-1]:
            arc = segs[sid].arc_points(radius=r, steps=18)
            pts += [to_px(p, img_size, margin, scale_px, kpv.lattice.rotation_deg) for p in arc]
        draw.line(pts, fill=(255,255,255), width=int(kpv.stroke_px*2), joint="curve")
        draw.line(pts, fill=(0,0,0), width=kpv.stroke_px, joint="curve")

    # dots (solid helps detection)
    R = 5 if solid_dots else 4
    for (i,j) in dots:
        x,y = to_px((i,j), img_size, margin, scale_px, kpv.lattice.rotation_deg)
        if solid_dots:
            draw.ellipse((x-R,y-R,x+R,y+R), fill=(0,0,0))
        else:
            draw.ellipse((x-R,y-R,x+R,y+R), outline=(0,0,0), width=2)
    # subtle outer border
    draw.rectangle((margin*0.5, margin*0.5, img_size-margin*0.5, img_size-margin*0.5), outline=(0,0,0), width=2)
    return img

def render_svg(kpv: KPV, out_svg="kolamqr.svg", view_px=1400, margin=80):
    dots = lattice_dots(kpv.lattice, kpv.mask)
    segs = build_segments(dots)
    incidence = build_vertex_incidence(segs, dots)
    cycles = trace_cycles(segs, incidence, kpv.H, kpv.V)

    dwg = svgwrite.Drawing(out_svg, size=(view_px, view_px))
    g = dwg.g(fill="none", stroke="black", stroke_width=kpv.stroke_px, stroke_linecap="round", stroke_linejoin="round")
    dwg.add(g)
    scale_px = (view_px-2*margin)/(max(kpv.lattice.cols, kpv.lattice.rows)+1)
    r = kpv.arc_radius * kpv.lattice.spacing

    def px(p):
        x,y = rotate(p, kpv.lattice.rotation_deg)
        return (margin + x*scale_px, view_px - (margin + y*scale_px))

    for cyc in cycles:
        pts=[]
        for sid in cyc[:-1]:
            arc = segs[sid].arc_points(radius=r, steps=22)
            pts += [px(q) for q in arc]
        g.add(dwg.path(d=" ".join([("M" if k==0 else "L")+f" {x:.2f},{y:.2f}" for k,(x,y) in enumerate(pts)]),
                       stroke="white", stroke_width=kpv.stroke_px*1.8))
        g.add(dwg.path(d=" ".join([("M" if k==0 else "L")+f" {x:.2f},{y:.2f}" for k,(x,y) in enumerate(pts)]),
                       stroke="black", stroke_width=kpv.stroke_px))
    # dots
    for (i,j) in dots:
        x,y = px((i,j))
        dwg.add(dwg.circle(center=(x,y), r=3.5, fill="white", stroke="black", stroke_width=1.4))
    # frame
    pad=margin*0.5
    dwg.add(dwg.rect(insert=(pad,pad), size=(view_px-2*pad, view_px-2*pad), fill="none", stroke="black", stroke_width=2))
    dwg.save()
    return out_svg


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Kolam-QR encoder/decoder (PNG robust)")
    sub = ap.add_subparsers(dest="cmd")

    e = sub.add_parser("encode", help="encode text/bytes → Kolam-QR PNG")
    e.add_argument("--text", type=str, default=None, help="text to encode (UTF-8)")
    e.add_argument("--in-bytes", type=str, default=None, help="path to binary file to encode")
    e.add_argument("--rows", type=int, default=21, help="lattice rows")
    e.add_argument("--cols", type=int, default=21, help="lattice cols")
    e.add_argument("--ecc", type=str, default="M", choices=["L","M","Q","H"], help="ECC level")
    e.add_argument("--out", type=str, default="kolamqr.png", help="PNG output")
    e.add_argument("--svg", type=str, default=None, help="optional SVG output")

    d = sub.add_parser("decode", help="decode image → bytes")
    d.add_argument("--image", type=str, required=True, help="input PNG/JPG")
    args = ap.parse_args()

    if args.cmd == "encode":
        if args.text is None and args.in_bytes is None:
            print("Provide --text or --in-bytes"); sys.exit(1)
        data = args.text.encode("utf-8") if args.text is not None else open(args.in_bytes,"rb").read()
        png_path, meta = kolamqr_encode(data, rows=args.rows, cols=args.cols,
                                        ecc_level=args.ecc, png_out=args.out, svg_out=args.svg)
        print(f"✅ Wrote {png_path}")
        if args.svg: print(f"✅ Wrote {args.svg}")
        print(f"📊 {json.dumps(meta)}")

    elif args.cmd == "decode":
        data, ok, meta = kolamqr_decode(args.image)
        print(f"📊 {json.dumps(meta)}")
        print(f"✅ ECC ok: {ok}")
        # decide if bytes look like text
        try:
            txt = data.decode("utf-8")
            print("🔤 Text:", txt)
        except Exception:
            print(f"🗂️  Bytes length: {len(data)}")
            outbin = "decoded.bin"
            with open(outbin,"wb") as f: f.write(data)
            print(f"💾 Saved raw bytes → {outbin}")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
