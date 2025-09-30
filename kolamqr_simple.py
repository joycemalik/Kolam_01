# kolamqr_simple.py
# Minimal, clean Kolam-QR (simple sikku look): single stroke, dot rings, anti-aliased PNG.

import os, sys, math, json, argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import cv2
from skimage.morphology import skeletonize

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
    kind: str          # "square"
    rows: int
    cols: int
    spacing: float = 1.0
    rotation_deg: float = 0.0

@dataclass
class KPV:
    lattice: Lattice
    mask: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    arc_radius: float = 0.48
    stroke_px: int = 7
    supersample: int = 2  # Reduced from 4 to 2 for better performance
    color: str = "white"      # "white" or hex like "#0a8a6d"
    bg: str = "dark"          # "light" or "dark"
    dot_ring_px: int = 4

def rotate(p, deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    x, y = p
    return (c*x - s*y, s*x + c*y)

def lattice_dots(latt: Lattice, mask: Optional[np.ndarray]) -> Set[Tuple[int,int]]:
    dots = {(i,j) for j in range(latt.rows) for i in range(latt.cols)}
    if mask is not None:
        dots = {(i,j) for (i,j) in dots if 0<=j<mask.shape[0] and 0<=i<mask.shape[1] and mask[j,i]==1}
    return dots


# =========================
# Segments + tracing
# =========================
MID_OFF = {'E':(0.5,0.0), 'N':(0.0,0.5), 'W':(-0.5,0.0), 'S':(0.0,-0.5)}
SEG_DEF = [('E','N'), ('N','W'), ('W','S'), ('S','E')]  # CCW quarters

class Segment:
    def __init__(self, seg_id, dot, a, b):
        self.id = seg_id
        self.dot = dot
        self.a = a; self.b = b

    def arc_points(self, radius=0.48, steps=12):  # Reduced default steps from 20 to 12
        (i,j) = self.dot
        cx, cy = (i, j)
        ang = {'E':0, 'N':90, 'W':180, 'S':270}
        start = ang[self.a]; end = (ang[self.b]) % 360
        if (end - start) % 360 != 90:   # force quarter CCW
            end = (start + 90) % 360
        A = np.linspace(math.radians(start), math.radians(end), steps)
        return [(cx + radius*math.cos(a), cy + radius*math.sin(a)) for a in A]

def build_segments(dots: Set[Tuple[int,int]]) -> List[Segment]:
    segs=[]; sid=0
    for (i,j) in dots:
        for (a,b) in SEG_DEF:
            segs.append(Segment(sid,(i,j),a,b)); sid+=1
    return segs

def midpoint_xy(dot, end_label):
    (i,j)=dot; dx,dy = MID_OFF[end_label]
    return (i+dx, j+dy)

def vertex_key(xy):
    return (round(xy[0]*2)/2.0, round(xy[1]*2)/2.0)

END_DIRS={'E':('N','S'),'W':('N','S'),'N':('E','W'),'S':('E','W')}

def build_vertex_incidence(segs: List[Segment], dots: Set[Tuple[int,int]]):
    inc: Dict[Tuple[float,float], List[Tuple[int,Tuple[int,int],str,str]]] = {}
    for s in segs:
        for end in [s.a, s.b]:
            v = vertex_key(midpoint_xy(s.dot, end))
            if end == s.a:
                dtag = END_DIRS[end][0] if (end in ('E','W') and s.b=='N') or (end in ('N','S') and s.b=='E') else END_DIRS[end][1]
            else:
                dtag = END_DIRS[end][1] if (end in ('E','W') and s.a=='N') or (end in ('N','S') and s.a=='E') else END_DIRS[end][0]
            inc.setdefault(v, []).append((s.id, s.dot, end, dtag))
    return inc

def pairings_at_vertex(vitems, H, V):
    if len(vitems)<=2:
        return [(vitems[0][0], vitems[1][0])] if len(vitems)==2 else []
    dots = list({it[1] for it in vitems})
    d1,d2 = dots[0], dots[1]
    if d1[1]==d2[1]:  # horizontal neighbors
        j=d1[1]; i=min(d1[0], d2[0])
        merge = int(H[j,i]) if 0<=j<H.shape[0] and 0<=i<H.shape[1] else 0
        Ns=[it for it in vitems if it[3]=='N']; Ss=[it for it in vitems if it[3]=='S']
        if merge==1 and len(Ns)==2 and len(Ss)==2:
            return [(Ns[0][0],Ns[1][0]),(Ss[0][0],Ss[1][0])]
        groups={}; [groups.setdefault(it[1],[]).append(it[0]) for it in vitems]
        return [(v[0],v[1]) for v in groups.values() if len(v)==2]
    else:            # vertical neighbors
        i=d1[0] if d1[0]==d2[0] else min(d1[0], d2[0]); j=min(d1[1], d2[1])
        merge = int(V[j,i]) if 0<=j<V.shape[0] and 0<=i<V.shape[1] else 0
        Es=[it for it in vitems if it[3]=='E']; Ws=[it for it in vitems if it[3]=='W']
        if merge==1 and len(Es)==2 and len(Ws)==2:
            return [(Es[0][0],Es[1][0]),(Ws[0][0],Ws[1][0])]
        groups={}; [groups.setdefault(it[1],[]).append(it[0]) for it in vitems]
        return [(v[0],v[1]) for v in groups.values() if len(v)==2]

def trace_cycles(segs: List[Segment], incidence, H, V) -> List[List[int]]:
    pair_map: Dict[int, List[int]] = {}
    for _, items in incidence.items():
        for a,b in pairings_at_vertex(items,H,V):
            pair_map.setdefault(a,[]).append(b)
            pair_map.setdefault(b,[]).append(a)
    visited=set(); cycles=[]
    for s in segs:
        if s.id in visited: continue
        path=[]; cur=s.id; prev=None
        while True:
            path.append(cur); visited.add(cur)
            nxts=[x for x in pair_map.get(cur,[]) if x!=prev]
            if not nxts: break
            prev,cur = cur,nxts[0]
            if cur==path[0]:
                path.append(cur); cycles.append(path); break
    uniq=[]; seen=set()
    for c in cycles:
        key=tuple(sorted(c))
        if key not in seen: seen.add(key); uniq.append(c)
    return uniq


# =========================
# QR-like reserves/mask/ECC
# =========================
def reserve_maps(rows, cols):
    H_res = np.zeros((rows, max(cols-1,1)), dtype=bool)
    V_res = np.zeros((max(rows-1,1), cols), dtype=bool)

    def reserve_block(x0,y0,w=3,h=3):
        for j in range(y0, y0+h):
            for i in range(x0, x0+w-1):
                if 0<=j<H_res.shape[0] and 0<=i<H_res.shape[1]: H_res[j,i]=True
        for j in range(y0, y0+h-1):
            for i in range(x0, x0+w):
                if 0<=j<V_res.shape[0] and 0<=i<V_res.shape[1]: V_res[j,i]=True

    # finders: TL, TR, BL
    reserve_block(0,0,3,3); reserve_block(cols-3,0,3,3); reserve_block(0,rows-3,3,3)

    trow = 3 if rows>4 else 1
    tcol = 3 if cols>4 else 1
    for i in range(H_res.shape[1]): H_res[trow,i]=True
    for j in range(V_res.shape[0]): V_res[j,tcol]=True

    fmt=[]
    for i in range(min(cols-1,40)): H_res[1,i]=True; fmt.append(('H',1,i))
    for j in range(min(rows-1,10)): V_res[j,1]=True; fmt.append(('V',j,1))
    return H_res, V_res, fmt, trow, tcol

def apply_finders_timing(H,V,trow,tcol):
    for i in range(H.shape[1]): H[trow,i]= 1 if (i%2==0) else 0
    for j in range(V.shape[0]): V[j,tcol]= 1 if (j%2==0) else 0
    # flowerish finders (merge inside)
    def fill_block(H,V,x0,y0):
        for j in range(y0,y0+3):
            for i in range(x0,x0+2):
                if 0<=j<H.shape[0] and 0<=i<H.shape[1]: H[j,i]=1
        for j in range(y0,y0+2):
            for i in range(x0,x0+3):
                if 0<=j<V.shape[0] and 0<=i<V.shape[1]: V[j,i]=1
        if 0<=y0<H.shape[0] and 0<=x0<H.shape[1]: H[y0,x0]=0
    fill_block(H,V,0,0); fill_block(H,V,H.shape[1]-2,0); fill_block(H,V,0,V.shape[0]-2)

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
        lambda i,j: ((i+j)%2)==0,
        lambda i,j: (j%2)==0,
        lambda i,j: (i%3)==0,
        lambda i,j: (((i*j)%2)+((i*j)%3))%2==0,
    ]

def apply_mask_bits(bits, indices, mask_id):
    mf = mask_functions()[mask_id%4]
    out=[]
    for (coord,b) in zip(indices,bits):
        _,j,i = coord
        out.append(b ^ int(mf(i,j)))
    return out

def rs_encode_blocks(data_bytes, ecc_bytes):
    if not RS_OK or ecc_bytes<=0: return data_bytes
    nsize=255; nsym=min(ecc_bytes,254); k=nsize-nsym
    rs=reedsolo.RSCodec(nsym, nsize=nsize)
    out=bytearray()
    for i in range(0,len(data_bytes),k):
        out.extend(rs.encode(bytearray(data_bytes[i:i+k])))
    return bytes(out)

def rs_decode_blocks(all_bytes, ecc_bytes):
    if not RS_OK or ecc_bytes<=0: return all_bytes, False
    nsize=255; nsym=min(ecc_bytes,254); k=nsize-nsym
    rs=reedsolo.RSCodec(nsym, nsize=nsize)
    out=bytearray(); ok=True
    for i in range(0,len(all_bytes),nsize):
        blk=bytearray(all_bytes[i:i+nsize])
        if len(blk)==0: continue
        try:
            dec=rs.decode(blk)
            out.extend(dec)
        except Exception:
            ok=False; out.extend(blk[:min(len(blk),k)])
    return bytes(out), ok


# =========================
# Auto-sizing helpers
# =========================
def _capacity_bits_for(rows: int, cols: int) -> int:
    # build reserves and count free H/V midpoints
    H_res, V_res, _, _, _ = reserve_maps(rows, cols)
    H_free = np.count_nonzero(~H_res)
    V_free = np.count_nonzero(~V_res)
    return H_free + V_free

def _rate_for_level(level: str) -> float:
    return {'L':0.10, 'M':0.18, 'Q':0.25, 'H':0.33}.get(level, 0.18)

def pick_lattice_for_payload(payload_len: int, ecc_level: str,
                             min_side: int = 17, max_side: int = 81,
                             step: int = 4, square: bool = True) -> Tuple[int,int, dict]:
    """
    Try odd-ish sizes: 17,21,25,... up to max_side (QR-style growth).
    Returns (rows, cols, info)
    """
    rate = _rate_for_level(ecc_level)
    candidates = []
    side = min_side
    while side <= max_side:
        rows = cols = side if square else side  # extend here if you want rectangular
        cap_bits = _capacity_bits_for(rows, cols)
        k_bytes = max(1, cap_bits // 8)
        ecc_bytes = max(1, min(int(rate * k_bytes), k_bytes - 1, 200))
        data_bytes = k_bytes - ecc_bytes
        fits = (data_bytes >= payload_len)
        candidates.append((rows, cols, cap_bits, k_bytes, ecc_bytes, data_bytes, fits))
        if fits:
            info = {
                "rows": rows, "cols": cols, "capacity_bits": cap_bits,
                "k_bytes": k_bytes, "ecc_bytes": ecc_bytes, "data_bytes": data_bytes
            }
            return rows, cols, info
        side += step
    # if nothing fits, take the largest and tell the caller
    rows, cols, cap_bits, k_bytes, ecc_bytes, data_bytes, _ = candidates[-1]
    info = {
        "rows": rows, "cols": cols, "capacity_bits": cap_bits,
        "k_bytes": k_bytes, "ecc_bytes": ecc_bytes, "data_bytes": data_bytes
    }
    return rows, cols, info


# =========================
# Encode
# =========================
def kolamqr_encode(data: bytes, rows: int=None, cols: int=None, ecc_level: str='M',
                   png_out: str="kolamqr.png", stroke_px=7, color="white", bg="dark", 
                   fast_mode=False, out_size=1200, supersample: int=2):
    # auto-pick lattice if needed
    if rows is None or cols is None:
        rows, cols, info = pick_lattice_for_payload(len(data), ecc_level)
    else:
        # still compute to report meta later
        cap_bits = _capacity_bits_for(rows, cols)
        k_bytes = max(1, cap_bits // 8)
        ecc_bytes = max(1, min(int(_rate_for_level(ecc_level)*k_bytes), k_bytes-1, 200))
        info = {"rows":rows,"cols":cols,"capacity_bits":cap_bits,
                "k_bytes":k_bytes,"ecc_bytes":ecc_bytes,"data_bytes":k_bytes-ecc_bytes}

    mask = np.ones((rows, cols), dtype=np.uint8)
    _ = lattice_dots(Lattice('square', rows, cols), mask)
    H = np.zeros((rows, max(cols-1,1)), dtype=np.uint8)
    V = np.zeros((max(rows-1,1), cols), dtype=np.uint8)

    H_res, V_res, fmt_slots, trow, tcol = reserve_maps(rows, cols)
    apply_finders_timing(H,V,trow,tcol)

    H_idx, V_idx = zigzag_indices(H_res, V_res)
    capacity_bits = len(H_idx)+len(V_idx)
    k_bytes = max(1, capacity_bits//8)
    rate = {'L':0.10,'M':0.18,'Q':0.25,'H':0.33}[ecc_level]
    ecc_bytes = int(rate*k_bytes)
    ecc_bytes = max(1, min(ecc_bytes, k_bytes-1, 200))
    data_bytes = k_bytes - ecc_bytes

    payload = data[:data_bytes]
    coded = rs_encode_blocks(payload, ecc_bytes)[:k_bytes]

    # bits
    bits=[]
    for b in coded:
        for t in range(8): bits.append((b>>(7-t))&1)
    bits = bits[:capacity_bits]

    # split
    nH=len(H_idx); bits_H=bits[:nH]; bits_V=bits[nH:]

    # mask choice
    best_id=0; bestH=bits_H; bestV=bits_V; best_score=None
    for mid in range(4):
        mH=apply_mask_bits(bits_H,H_idx,mid)
        mV=apply_mask_bits(bits_V,V_idx,mid)
        score = abs(sum(mH)+sum(mV) - 0.5*(len(mH)+len(mV)))
        if best_score is None or score<best_score:
            best_score=score; best_id=mid; bestH, bestV = mH, mV

    for (coord,b) in zip(H_idx, bestH): _,j,i=coord; H[j,i]=b
    for (coord,b) in zip(V_idx, bestV): _,j,i=coord; V[j,i]=b

    # format bits
    def tb(val,n): return [ (val>>(n-1-k))&1 for k in range(n) ]
    ecc2={'L':0,'M':1,'Q':2,'H':3}[ecc_level]
    fmt_bits = tb(best_id,3)+tb(ecc2,2)+tb(rows,6)+tb(cols,6)
    x=0
    for b in fmt_bits: x=((x<<1)&0xFF) ^ (0x07 if ((x>>7)&1)^b else 0)
    fmt_bits += tb(x,8)
    for (slot,b) in zip(fmt_slots, fmt_bits):
        t,j,i = slot
        if t=='H': H[j,i]=b
        else: V[j,i]=b

    # render with performance optimizations
    use_supersample = 1 if fast_mode else supersample
    kpv = KPV(
        lattice=Lattice('square', rows, cols),
        mask=mask, H=H, V=V,
        arc_radius=0.48, stroke_px=stroke_px,
        supersample=use_supersample, color=color, bg=bg, dot_ring_px=4
    )
    img = render_simple_png(kpv, out_size=out_size, margin=90)
    img.save(png_out)
    
    # Use info from auto-sizing if available, otherwise compute fresh meta
    if 'data_bytes' in info:
        meta = dict(info)
        meta.update({"mask_id":best_id,"ecc":ecc_level})
    else:
        meta = {"rows":rows,"cols":cols,"capacity_bits":capacity_bits,
                "k_bytes":k_bytes,"ecc_bytes":ecc_bytes,"mask_id":best_id,"ecc":ecc_level}
    
    return png_out, meta


# =========================
# Decode
# =========================
def detect_dots(gray: np.ndarray) -> np.ndarray:
    # Downsample large images for faster processing
    h, w = gray.shape
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (new_w, new_h))
        scale_factor = max(h, w) / 1500
    else:
        scale_factor = 1.0
    
    g = cv2.GaussianBlur(gray,(5,5),0)
    th = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,7)
    
    # Try multiple parameter sets for blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea=True; params.minArea=4; params.maxArea=10000
    params.filterByCircularity=False; params.filterByInertia=False; params.filterByConvexity=False
    params.filterByColor=False
    detector=cv2.SimpleBlobDetector_create(params)
    kps=detector.detect(th)
    
    # If not enough blobs, try with inverted threshold
    if len(kps) < 9:
        th_inv = cv2.bitwise_not(th)
        kps = detector.detect(th_inv)
    
    # Scale points back up if we downsampled
    pts = np.array([[kp.pt[0] * scale_factor, kp.pt[1] * scale_factor] for kp in kps], dtype=np.float32)
    return pts

def fit_lattice_from_dots(pts: np.ndarray):
    if pts.shape[0]<9: raise RuntimeError("Not enough dots")
    X = pts - pts.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt; B = X @ R.T
    def step(c):
        v=np.sort(B[:,c]); d=np.diff(v); d=d[d>1e-3]
        return np.median(d) if len(d)>0 else 0
    sx,sy = step(0), step(1)
    if sx<=0 or sy<=0: raise RuntimeError("Bad lattice fit")
    G = np.stack([np.round(B[:,0]/sx), np.round(B[:,1]/sy)], axis=1).astype(int)
    G[:,0]-=G[:,0].min(); G[:,1]-=G[:,1].min()
    rows=int(G[:,1].max()+1); cols=int(G[:,0].max()+1)
    mask=np.zeros((rows,cols),dtype=np.uint8)
    for gx,gy in G:
        if 0<=gy<rows and 0<=gx<cols: mask[gy,gx]=1
    return (R, pts.mean(0), sx, sy, rows, cols, mask)

def recover_HV_from_image(gray: np.ndarray, R, mean, sx, sy, rows, cols):
    # Downsample for faster skeletonization if image is large
    h, w = gray.shape
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        gray_small = cv2.resize(gray, (new_w, new_h))
        scale_factor = max(h, w) / 1500
    else:
        gray_small = gray
        scale_factor = 1.0
    
    g = cv2.GaussianBlur(gray_small,(3,3),0)
    thr = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    inv = (cv2.bitwise_not(thr)>0).astype(np.uint8)
    skel = skeletonize(inv).astype(np.uint8)

    def lattice_to_px(gx, gy):
        b=np.array([gx*sx, gy*sy],dtype=np.float32)
        p=b@R + mean
        # Scale down if we're working with downsampled image
        return int(round(p[0] / scale_factor)), int(round(p[1] / scale_factor))

    H=np.zeros((rows,max(cols-1,1)),dtype=np.uint8)
    V=np.zeros((max(rows-1,1),cols),dtype=np.uint8)

    def sample_dir(x,y,dx,dy,L=4):  # Reduced sampling length
        s=0
        for t in range(-L,L+1):
            xi=int(round(x+t*dx)); yi=int(round(y+t*dy))
            if 0<=yi<skel.shape[0] and 0<=xi<skel.shape[1]:
                s+=skel[yi,xi]
        return s

    for j in range(rows):
        for i in range(cols-1):
            x,y = lattice_to_px(i+0.5,j)
            vert=sample_dir(x,y,0,1); horz=sample_dir(x,y,1,0)
            H[j,i]=1 if vert>=horz else 0

    for j in range(rows-1):
        for i in range(cols):
            x,y = lattice_to_px(i,j+0.5)
            horz=sample_dir(x,y,1,0); vert=sample_dir(x,y,0,1)
            V[j,i]=1 if horz>=vert else 0
    return H,V

def parse_format(H,V,rows,cols):
    H_res,V_res,fmt,trow,tcol = reserve_maps(rows,cols)
    bits=[]
    for t,j,i in fmt: bits.append( H[j,i] if t=='H' else V[j,i] )
    if len(bits)<25: raise RuntimeError("format missing")
    mask_id = (bits[0]<<2)|(bits[1]<<1)|bits[2]
    ecc2 = (bits[3]<<1)|bits[4]
    ecc = {0:'L',1:'M',2:'Q',3:'H'}.get(ecc2,'M')
    return mask_id, ecc

def kolamqr_decode(image_path: str):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None: raise FileNotFoundError(image_path)
    pts = detect_dots(gray)
    R, mean, sx, sy, rows, cols, mask = fit_lattice_from_dots(pts)
    H,V = recover_HV_from_image(gray, R, mean, sx, sy, rows, cols)
    mask_id, ecc = parse_format(H,V,rows,cols)

    H_res,V_res,fmt,trow,tcol = reserve_maps(rows,cols)
    H_idx,V_idx = zigzag_indices(H_res,V_res)
    capacity_bits = len(H_idx)+len(V_idx)
    k_bytes = max(1, capacity_bits//8)
    rate = {'L':0.10,'M':0.18,'Q':0.25,'H':0.33}[ecc]
    ecc_bytes = int(rate*k_bytes); ecc_bytes=max(1,min(ecc_bytes,k_bytes-1,200))

    mH=[H[j,i] for (_,j,i) in H_idx]; mV=[V[j,i] for (_,j,i) in V_idx]
    bits_H = apply_mask_bits(mH,H_idx,mask_id)
    bits_V = apply_mask_bits(mV,V_idx,mask_id)
    bits_all = bits_H + bits_V
    total_bytes = len(bits_all)//8
    b=[]
    for k in range(total_bytes):
        v=0
        for t in range(8): v=(v<<1)|bits_all[k*8+t]
        b.append(v)
    b=bytes(b)[:k_bytes]
    data, ok = rs_decode_blocks(b, ecc_bytes)
    return data, ok, {"rows":rows,"cols":cols,"mask_id":mask_id,"ecc":ecc}


# =========================
# Simple kolam renderer (pretty but minimal)
# =========================
def render_simple_png(kpv: KPV, out_size=1200, margin=90) -> Image.Image:
    rows, cols = kpv.lattice.rows, kpv.lattice.cols
    dots = lattice_dots(kpv.lattice, kpv.mask)
    segs = build_segments(dots)
    incidence = build_vertex_incidence(segs, dots)
    cycles = trace_cycles(segs, incidence, kpv.H, kpv.V)

    # Reduce supersample for performance - still good quality
    SS = max(1, min(2, kpv.supersample))  # Cap at 2x instead of 4x
    W = out_size*SS; Hh = out_size*SS; M = margin*SS
    scale_px = (out_size*SS - 2*M) / (max(cols, rows) + 1)
    arc_r = kpv.arc_radius * kpv.lattice.spacing

    # background
    if kpv.bg == "dark":
        bg = (45,33,26)          # dark brown (chalk board)
        line = (245,245,245) if kpv.color=="white" else (10,138,109)  # white or green
        dot_ring = (245,245,245)
    else:
        bg = (245,245,245)
        line = (16,120,96) if kpv.color!="white" else (30,30,30)
        dot_ring = (30,30,30)

    img = Image.new("RGB", (W,Hh), bg)
    draw = ImageDraw.Draw(img)

    def px(p):
        x,y = rotate(p, kpv.lattice.rotation_deg)
        return (M + x*scale_px, Hh - (M + y*scale_px))

    # Pre-calculate all points to avoid repeated calculations
    all_cycle_points = []
    for cyc in cycles:
        pts=[]
        for sid in cyc[:-1]:
            # Reduce arc steps for performance - 12 is still smooth enough
            for q in segs[sid].arc_points(radius=arc_r, steps=12):
                pts.append(px(q))
        if len(pts)>=2:
            all_cycle_points.append(pts)

    # Single pass drawing - remove expensive blur operation
    for pts in all_cycle_points:
        # Draw with slightly wider line for anti-aliasing effect
        draw.line(pts, fill=line, width=kpv.stroke_px*SS+1, joint="curve")

    # tiny rings on dots (very traditional)
    ring = max(1*SS, kpv.dot_ring_px*SS)
    for (i,j) in dots:
        x,y = px((i,j))
        draw.ellipse((x-ring,y-ring,x+ring,y+ring), outline=dot_ring, width=max(1,SS//2))

    # outer frame
    draw.rectangle((M*0.6, M*0.6, W-M*0.6, Hh-M*0.6), outline=dot_ring, width=max(1,SS//2))

    # downsample → crisp anti-aliased (only if supersampled)
    if SS > 1:
        final = img.resize((out_size,out_size), Image.Resampling.LANCZOS)
        return final
    else:
        return img


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Kolam-QR (simple sikku look) — PNG encode/decode")
    sub = ap.add_subparsers(dest="cmd")

    e = sub.add_parser("encode", help="encode text/bytes → Kolam-QR PNG")
    e.add_argument("--text", type=str, default=None)
    e.add_argument("--in-bytes", type=str, default=None)
    e.add_argument("--rows", type=int, default=None, help="lattice rows (auto if omitted)")
    e.add_argument("--cols", type=int, default=None, help="lattice cols (auto if omitted)")
    e.add_argument("--ecc", type=str, default="M", choices=["L","M","Q","H"])
    e.add_argument("--out", type=str, default="kolamqr.png")
    e.add_argument("--stroke", type=int, default=7, help="line thickness")
    e.add_argument("--color", type=str, default="white", help="'white' or hex like '#0a8a6d'")
    e.add_argument("--bg", type=str, default="dark", choices=["dark","light"])
    e.add_argument("--fast", action="store_true", help="fast mode - lower quality but much faster")
    e.add_argument("--size", type=int, default=1200, help="output image size in pixels")
    e.add_argument("--auto", action="store_true", help="force auto size picking")
    e.add_argument("--ss", type=int, default=2, help="supersample factor (2=fast)")

    d = sub.add_parser("decode", help="decode PNG/JPG → bytes/text")
    d.add_argument("--image", type=str, required=True)

    args = ap.parse_args()
    if args.cmd=="encode":
        if args.text is None and args.in_bytes is None:
            print("Provide --text or --in-bytes"); sys.exit(1)
        data = args.text.encode("utf-8") if args.text is not None else open(args.in_bytes,"rb").read()
        
        # Force auto-sizing if --auto is specified
        rows = None if args.auto else args.rows
        cols = None if args.auto else args.cols
        
        out, meta = kolamqr_encode(
            data, rows=rows, cols=cols, ecc_level=args.ecc,
            png_out=args.out, stroke_px=args.stroke, color=args.color, bg=args.bg,
            fast_mode=args.fast, out_size=args.size, supersample=args.ss
        )
        print(f"✅ wrote {out}")
        print("📊", json.dumps(meta))
        if meta.get("data_bytes", 0) < len(data):
            print("⚠️ message truncated to fit this version. Use --auto or larger rows/cols.")
    elif args.cmd=="decode":
        data, ok, meta = kolamqr_decode(args.image)
        print("📊", json.dumps(meta))
        print("✅ ECC ok:", ok)
        try:
            print("🔤", data.decode("utf-8"))
        except:
            print("🗂 bytes:", len(data))
            with open("decoded.bin","wb") as f: f.write(data)
            print("💾 saved → decoded.bin")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
