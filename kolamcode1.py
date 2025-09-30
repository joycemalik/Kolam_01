import argparse, base64, binascii, json, math, os, sys
from typing import Tuple, List
import numpy as np
import cv2
import svgwrite
from xml.etree import ElementTree as ET
from PIL import Image
import hashlib
from math import cos, sin, pi

# Optional imports for SVG support
CAIROSVG_AVAILABLE = False
SVGLIB_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except (ImportError, OSError) as e:
    # OSError can occur if cairo libraries are missing
    CAIROSVG_AVAILABLE = False

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    SVGLIB_AVAILABLE = True
except ImportError:
    SVGLIB_AVAILABLE = False

def get_supported_formats() -> List[str]:
    """Return list of supported image formats"""
    basic_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif']
    svg_formats = ['.svg']
    
    # Check if SVG support is available
    svg_available = CAIROSVG_AVAILABLE or SVGLIB_AVAILABLE
    
    if svg_available:
        return basic_formats + svg_formats
    else:
        return basic_formats

def print_format_info():
    """Print information about supported formats"""
    formats = get_supported_formats()
    print("Supported image formats:")
    for fmt in formats:
        print(f"  ✅ {fmt}")
    
    if '.svg' not in formats:
        print(f"\n❌ SVG support not available")
        print("For SVG support, install one of:")
        print("  • pip install cairosvg")
        print("  • pip install svglib reportlab")
    else:
        svg_backend = "cairosvg" if CAIROSVG_AVAILABLE else "svglib+reportlab"
        print(f"\n✅ SVG support available via {svg_backend}")

# =========================
# Utility: bits <-> bytes
# =========================
def bytes_to_bits(data: bytes) -> List[int]:
    out = []
    for b in data:
        for i in range(8):
            out.append((b >> (7 - i)) & 1)
    return out

def bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits + [0] * (8 - (len(bits) % 8))
    by = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] & 1)
        by.append(byte)
    return bytes(by)

def crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xffffffff

# =========================
# Layout constants
# =========================
# We keep a quiet margin where finder circles and border live.
# Grid: N x N tiles. Each tile encodes 3 bits: rot(2) + dot(1).

def pack_header(n_tiles: int, text_len: int, version: int = 1) -> bytes:
    """
    Header: 1 byte version | 2 bytes N (grid) | 4 bytes text_len | 4 bytes CRC32(payload)
    We'll put CRC over the raw text bytes (payload).
    """
    if n_tiles > 65535 or text_len > 0xFFFFFFFF:
        raise ValueError("Too big.")
    return bytes([
        version,
        (n_tiles >> 8) & 0xFF, n_tiles & 0xFF
    ]) + text_len.to_bytes(4, 'big')

def unpack_header(h: bytes) -> Tuple[int, int, int]:
    if len(h) < 7:
        raise ValueError("Header too short")
    version = h[0]
    n_tiles = (h[1] << 8) | h[2]
    text_len = int.from_bytes(h[3:7], 'big')
    return version, n_tiles, text_len

# =========================
# Tile drawing (SVG)
# =========================
def draw_tile(dwg, group, x, y, s, rot, dotbit, stroke="black", stroke_width=2.0):
    """
    Kolam-like tile: two quarter arcs connecting midpoints.
    rot in {0,1,2,3}: rotation by 90*rot degrees.
    dotbit in {0,1}: if 1, place a small circle at center for the 3rd bit.
    """
    half = s / 2
    r = s / 2 * 0.95

    # midpoints in base orientation: left, top, right, bottom
    mids = [
        (x, y + half),
        (x + half, y),
        (x + s, y + half),
        (x + half, y + s)
    ]

    def rot_point(px, py, cx, cy, turns):
        turns %= 4
        dx, dy = px - cx, py - cy
        for _ in range(turns):
            dx, dy = -dy, dx
        return (cx + dx, cy + dy)

    cx, cy = x + half, y + half
    mids_rot = [rot_point(px, py, cx, cy, rot) for (px, py) in mids]

    def arc(p0, p1):
    # Manually construct the entire path 'd' string at once.
    # This avoids the buggy behavior of path.push() adding a leading space.
    # M = Move To, A = Elliptical Arc
        d_string = (f"M {p0[0]:.3f},{p0[1]:.3f} "
                f"A {r:.3f},{r:.3f} 0 0 1 {p1[0]:.3f},{p1[1]:.3f}")

    # Create the path element with the complete, valid 'd' string.
        path = dwg.path(d=d_string, fill="none", stroke=stroke,
                    stroke_width=stroke_width, stroke_linecap="round")
        group.add(path)

    # base: (left->top) and (right->bottom)
    arc(mids_rot[0], mids_rot[1])
    arc(mids_rot[2], mids_rot[3])

    if dotbit == 1:
        dot_r = s * 0.07
        group.add(dwg.circle(center=(cx, cy), r=dot_r, fill=stroke))


def draw_finders(dwg, canvas, margin, size, finder_r):
    # Three finder circles at TL, TR, BL (like QR)
    TL = (margin * 0.6, margin * 0.6)
    TR = (size - margin * 0.6, margin * 0.6)
    BL = (margin * 0.6, size - margin * 0.6)
    for c in [TL, TR, BL]:
        canvas.add(dwg.circle(center=c, r=finder_r, fill="black", stroke="none"))

def render_svg(grid_bits: List[int], N: int, tile_bits=3,
               out_svg="kolam.svg", tile_px=42, margin_px=180, stroke_px=3.0):
    # Compute needed tiles = N*N; pad bits
    total_tiles = N * N
    bits = grid_bits[:]
    if len(bits) < total_tiles * tile_bits:
        bits += [0] * (total_tiles * tile_bits - len(bits))
    elif len(bits) > total_tiles * tile_bits:
        bits = bits[: total_tiles * tile_bits]

    size = margin_px * 2 + N * tile_px
    dwg = svgwrite.Drawing(out_svg, size=(size, size))
    canvas = dwg.add(dwg.g())

    # Outer border
    canvas.add(dwg.rect(insert=(2,2), size=(size-4, size-4),
                        stroke="black", fill="none", stroke_width=4))

    # Finder markers
    finder_r = margin_px * 0.22
    draw_finders(dwg, canvas, margin_px, size, finder_r)

    # Tiles
    g_tiles = dwg.add(dwg.g())
    k = 0
    for r in range(N):
        for c in range(N):
            b0 = bits[k]; b1 = bits[k+1]; b2 = bits[k+2]
            k += 3
            rot = (b0 << 1) | b1
            dot = b2
            x = margin_px + c * tile_px
            y = margin_px + r * tile_px
            draw_tile(dwg, g_tiles, x, y, tile_px, rot, dot, stroke_width=stroke_px)

    dwg.save()
    return out_svg

# =========================
# Encoding: text -> bits
# =========================
def build_bitstream_from_text(text: str, N: int) -> List[int]:
    payload = text.encode("utf-8")
    h = pack_header(N, len(payload), version=1)
    chk = crc32(payload).to_bytes(4, 'big')
    full = h + payload + chk
    return bytes_to_bits(full)

# =========================
# Decoder helpers
# =========================
def find_border_alternative(gray: np.ndarray):
    """Alternative border detection for images without clear rectangular borders"""
    h, w = gray.shape
    
    # Try to find the content area by detecting the kolam pattern itself
    # Look for areas with high edge density (the kolam patterns)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find non-zero regions
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) == 0:
        return None
    
    # Get bounding box of all pattern content
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add some margin
    margin = min(w, h) * 0.05
    x_min = max(0, int(x_min - margin))
    y_min = max(0, int(y_min - margin))
    x_max = min(w-1, int(x_max + margin))
    y_max = min(h-1, int(y_max + margin))
    
    # Create artificial border points
    border_quad = np.array([
        [[x_min, y_min]],  # top-left
        [[x_max, y_min]],  # top-right
        [[x_max, y_max]],  # bottom-right
        [[x_min, y_max]]   # bottom-left
    ], dtype=np.int32)
    
    print(f"Using content-based border: ({x_min},{y_min}) to ({x_max},{y_max})")
    return border_quad

def largest_quad_contour(gray: np.ndarray):
    # Find the largest 4-point contour (outer border) with multiple methods
    h, w = gray.shape
    min_area = (h * w) * 0.1  # Border should be at least 10% of image
    
    # Method 1: Adaptive threshold
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    
    # Method 2: Otsu threshold
    _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 3: Manual threshold (for very clear borders)
    _, th3 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    best = None
    best_area = 0
    best_th = None
    
    # Try all three threshold methods
    for i, th in enumerate([th1, th2, th3]):
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            peri = cv2.arcLength(cnt, True)
            # Try different approximation tolerances
            for tolerance in [0.02, 0.03, 0.01, 0.05]:
                approx = cv2.approxPolyDP(cnt, tolerance * peri, True)
                if len(approx) == 4 and area > best_area:
                    best_area = area
                    best = approx
                    best_th = th
                    print(f"Found border using method {i+1}, tolerance {tolerance}, area {area}")
                    break
            if best is not None:
                break
        if best is not None:
            break
    
    # If no rectangular border found, try content-based detection
    if best is None:
        print("No rectangular border found, trying content-based detection...")
        best = find_border_alternative(gray)
        best_th = th1
    
    return best, best_th if best_th is not None else th1

def order_corners(pts: np.ndarray) -> np.ndarray:
    # Order 4 points: TL, TR, BR, BL
    pts = pts.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_to_square(img: np.ndarray, quad: np.ndarray, out_size: int):
    src = order_corners(quad)
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (out_size, out_size))
    return warped

# Build tile templates (binary) for rotation classification
def build_tile_templates(px: int, stroke: int=3) -> List[np.ndarray]:
    s = px
    img0 = np.zeros((s, s), np.uint8)
    # Two arcs: left->top and right->bottom (rot 0)
    half = s//2
    r = int(0.95 * s/2)
    # Simple raster draw using polylines approximating arcs
    def draw_arc(im, p0, p1):
        # approximate with 90-degree arc using cv2.ellipse
        center = (half, half)
        # We’ll draw four canonical arcs depending on endpoints
        # Build masks using cv2.ellipse around center
        pass

    # Simpler: draw with cv2.ellipse for four rotations directly
    def tile_image(rotation):
        im = np.zeros((s, s), np.uint8)
        # arc 1
        if rotation == 0:
            cv2.ellipse(im, (half, half), (r, r), 0, 180, 270, 255, thickness=stroke)
            cv2.ellipse(im, (half, half), (r, r), 0, 0, 90, 255, thickness=stroke)
        elif rotation == 1:
            cv2.ellipse(im, (half, half), (r, r), 0, 270, 360, 255, thickness=stroke)
            cv2.ellipse(im, (half, half), (r, r), 0, 90, 180, 255, thickness=stroke)
        elif rotation == 2:
            cv2.ellipse(im, (half, half), (r, r), 0, 0, 90, 255, thickness=stroke)
            cv2.ellipse(im, (half, half), (r, r), 0, 180, 270, 255, thickness=stroke)
        else:  # rotation == 3
            cv2.ellipse(im, (half, half), (r, r), 0, 90, 180, 255, thickness=stroke)
            cv2.ellipse(im, (half, half), (r, r), 0, 270, 360, 255, thickness=stroke)
        return im

    return [tile_image(0), tile_image(1), tile_image(2), tile_image(3)]

def classify_tile(tile_roi: np.ndarray, templates: List[np.ndarray]) -> Tuple[int, int]:
    """
    Return (rot, dotbit).
    We classify rotation by max normalized correlation with 4 templates.
    Dotbit: check small disk near center for filled pixels.
    """
    h, w = tile_roi.shape[:2]
    # binarize
    th = cv2.threshold(tile_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # resize to template size
    T = templates[0].shape[0]
    roi = cv2.resize(th, (T, T), interpolation=cv2.INTER_AREA)

    # rotation: normalized cross-correlation
    scores = []
    for t in templates:
        s = (t.astype(np.float32) / 255.0)
        r = (roi.astype(np.float32) / 255.0)
        num = np.sum(s * r)
        den = math.sqrt(np.sum(s*s) * np.sum(r*r) + 1e-6)
        scores.append(num / den)
    rot = int(np.argmax(scores))

    # dotbit: check center disk
    mask = np.zeros((T, T), np.uint8)
    cv2.circle(mask, (T//2, T//2), int(0.08*T), 255, -1)
    dot_pixels = np.sum(roi[mask==255]) / 255.0
    # threshold: center dot adds many white pixels
    dotbit = 1 if dot_pixels > (0.35 * np.sum(mask==255)) else 0
    return rot, dotbit

def load_image_universal(img_path: str) -> np.ndarray:
    """
    Load image from various formats: PNG, JPG, SVG, WEBP, etc.
    Returns grayscale numpy array.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    file_ext = os.path.splitext(img_path)[1].lower()
    
    try:
        if file_ext == '.svg':
            # Handle SVG files by converting to PNG first
            print(f"Converting SVG to image...")
            
            # Try cairosvg first (more reliable)
            if CAIROSVG_AVAILABLE:
                try:
                    png_data = cairosvg.svg2png(url=img_path)
                    img_array = np.frombuffer(png_data, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except Exception as svg_err:
                    print(f"cairosvg failed: {svg_err}")
                    img = None
            elif SVGLIB_AVAILABLE:
                try:
                    drawing = svg2rlg(img_path)
                    img_pil = renderPM.drawToPIL(drawing)
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except Exception as pil_err:
                    print(f"svglib failed: {pil_err}")
                    img = None
            else:
                raise RuntimeError("SVG support requires 'cairosvg' or 'svglib+reportlab'. Install with: pip install cairosvg")
            
            if img is None:
                raise RuntimeError("Failed to convert SVG to image")
        
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            # Handle standard image formats with OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                # Fallback to PIL for better format support
                pil_img = Image.open(img_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        else:
            # Try PIL for any other format
            print(f"Unknown format {file_ext}, trying PIL...")
            pil_img = Image.open(img_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        print(f"Successfully loaded {file_ext} image: {gray.shape}")
        return gray
        
    except Exception as e:
        raise RuntimeError(f"Error loading image {img_path}: {e}")

def read_grid_from_image(img_path: str, N_hint: int=None, debug: bool=False) -> Tuple[str, dict]:
    """
    Enhanced image reader supporting multiple formats
    """
    gray = load_image_universal(img_path)
    
    if debug:
        cv2.imwrite('debug_gray.png', gray)
        print(f"Saved debug_gray.png - Original grayscale image")

    quad, th = largest_quad_contour(gray)
    
    if debug:
        cv2.imwrite('debug_threshold.png', th)
        print(f"Saved debug_threshold.png - Threshold image")
        
        if quad is not None:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_img, [quad], -1, (0, 255, 0), 3)
            cv2.imwrite('debug_border.png', debug_img)
            print(f"Saved debug_border.png - Detected border")
    
    if quad is None:
        # Additional diagnostics
        print(f"Image shape: {gray.shape}")
        print(f"Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # Show what contours were found
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")
        for i, cnt in enumerate(contours[:5]):  # Show first 5
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(f"  Contour {i}: area={area:.0f}, vertices={len(approx)}")
        
        raise RuntimeError("Could not find outer border. Try:\n"
                          "1. Ensure the image has a clear black rectangular border\n"
                          "2. Increase image contrast\n"
                          "3. Use --debug flag to see detection images\n"
                          "4. Check that the border is at least 4 pixels thick")

    # Warp to square canvas
    OUT = 1600
    warped = warp_to_square(gray, quad, OUT)
    
    if debug:
        cv2.imwrite('debug_warped.png', warped)
        print(f"Saved debug_warped.png - Perspective corrected image")

    # Try to auto-estimate N if not known by finding finder circles spacing.
    # For MVP, we rely on metadata header inside the bitstream, so we just pick a plausible N for sampling,
    # then we’ll read header to get true N and re-sample if needed.
    # Sampling parameters:
    margin_px = int(OUT * 0.12)  # must match generator ratio
    # We'll try range of N values if N_hint not provided
    Ns = [N_hint] if N_hint else [21, 25, 29, 33, 37]
    templates = build_tile_templates(48, stroke=5)

    def sample_bits(N):
        tile_px = (OUT - 2 * margin_px) / N
        bits = []
        for r in range(N):
            for c in range(N):
                x0 = int(margin_px + c * tile_px)
                y0 = int(margin_px + r * tile_px)
                roi = warped[y0:y0+int(tile_px), x0:x0+int(tile_px)]
                if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
                    return None
                rot, dotbit = classify_tile(roi, templates)
                b0 = (rot >> 1) & 1
                b1 = rot & 1
                b2 = dotbit
                bits.extend([b0, b1, b2])
        return bits

    last_err = None
    for N in Ns:
        bits = sample_bits(N)
        if bits is None:
            continue
        # Try decode header
        header_bits = bits[:8*7]  # 7 bytes
        header = bits_to_bytes(header_bits)
        try:
            version, n_tiles, text_len = unpack_header(header)
        except Exception as e:
            last_err = e
            continue
        # Sanity: n_tiles should equal N*N
        if n_tiles != N*N or text_len < 0 or text_len > 200000:
            last_err = ValueError("Header sanity failed")
            continue

        # Rebuild full bytes and CRC
        total_bits_needed = 8*(7 + text_len + 4)
        if len(bits) < total_bits_needed:
            last_err = ValueError("Not enough bits captured")
            continue
        blob = bits_to_bytes(bits[:total_bits_needed])
        version2, n_tiles2, text_len2 = unpack_header(blob[:7])
        payload = blob[7:7+text_len2]
        crc_recv = int.from_bytes(blob[7+text_len2:7+text_len2+4], 'big')
        if crc_recv != crc32(payload):
            last_err = ValueError("CRC mismatch")
            continue

        text = payload.decode("utf-8", errors="replace")
        meta = {"version": version2, "N": N, "tiles": n_tiles2}
        return text, meta

    raise RuntimeError(f"Decode failed. Last error: {last_err}")

# =========================
# Main encode/decode
# =========================
def encode_cmd(args):
    text = args.text
    # auto grid if not provided
    N = args.grid if args.grid else auto_grid_for_text(len(text.encode("utf-8")), bits_per_tile=3)

    bits = bytes_to_bits(pack_header(N*N, len(text.encode('utf-8')))) \
           + bytes_to_bits(text.encode('utf-8')) \
           + bytes_to_bits(crc32(text.encode('utf-8')).to_bytes(4, 'big'))

    if args.style == "fancy":
        svg_path = render_svg_fancy(bits, N=N, tile_bits=3,
                                    out_svg=args.out, tile_px=args.tile_px,
                                    margin_px=args.margin_px, stroke_px=args.stroke_px,
                                    palette=args.palette, fractal_depth=args.fractal_depth,
                                    dot_opacity=args.dot_opacity)
    elif args.style == "clean":
        svg_path = render_svg_clean(bits, N=N, tile_bits=3,
                                    out_svg=args.out, tile_px=args.tile_px,
                                    margin_px=args.margin_px, stroke_px=args.stroke_px,
                                    dot_opacity=args.dot_opacity)
    else:
        svg_path = render_svg(bits, N=N, tile_bits=3,
                              out_svg=args.out, tile_px=args.tile_px,
                              margin_px=args.margin_px, stroke_px=args.stroke_px)

    print(f"Saved: {svg_path}")
    print(f"Grid: {N}x{N}, capacity bits: {N*N*3}, used bits: {len(bits)}")


def decode_cmd(args):
    # Validate input file format
    supported_formats = get_supported_formats()
    file_ext = os.path.splitext(args.image)[1].lower()
    
    if file_ext not in supported_formats:
        print(f"❌ Unsupported format: {file_ext}")
        print_format_info()
        return
    
    try:
        text, meta = read_grid_from_image(args.image, N_hint=args.grid, debug=args.debug)
        print("✅ Decoded text:")
        print(text)
        print(f"\n📊 Meta: {meta}")
        print(f"🎯 Image format: {file_ext}")
    except Exception as e:
        print(f"❌ Decode failed: {e}")
        if "SVG" in str(e):
            print("\n💡 For SVG support, install: pip install cairosvg")
        elif "border" in str(e).lower():
            print(f"\n💡 Try using --debug to see detection images")
            print(f"💡 Or use --help to see troubleshooting options")

def formats_cmd(args):
    print_format_info()

def build_parser():
    p = argparse.ArgumentParser(description="KolamCode: encode/decode with multi-format support")
    sub = p.add_subparsers()

    e = sub.add_parser("encode", help="encode text -> Kolam SVG")
    e.add_argument("--text", required=True, help="text to encode")
    e.add_argument("--out", default="kolam.svg", help="output SVG path")
    e.add_argument("--grid", type=int, default=None, help="grid size N (omit for auto)")
    e.add_argument("--tile-px", type=int, default=42, help="tile size in pixels")
    e.add_argument("--margin-px", type=int, default=180, help="quiet margin in pixels")
    e.add_argument("--stroke-px", type=float, default=3.0, help="(kept for legacy renderer)")
    e.add_argument("--style", choices=["basic","fancy","clean"], default="fancy", help="visual style")
    e.add_argument("--palette", choices=["basic","indigo","vermilion","jade","royal"], default="basic")
    e.add_argument("--dot-opacity", type=float, default=None, help="dot transparency (0.0-1.0, default varies by palette)")
    e.add_argument("--fractal-depth", type=int, default=1, help="0–2 tiny inner ropes")
    e.set_defaults(func=encode_cmd)



    d = sub.add_parser("decode", help="decode image -> text")
    d.add_argument("--image", required=True, help="input image (PNG/JPG/SVG/WEBP/BMP screenshot or photo)")
    d.add_argument("--grid", type=int, default=None, help="optional hint for N (e.g., 29)")
    d.add_argument("--debug", action="store_true", help="save debug images showing detection process")
    d.set_defaults(func=decode_cmd)
    
    f = sub.add_parser("formats", help="list supported image formats")
    f.set_defaults(func=formats_cmd)
    
    return p



def auto_grid_for_text(text_len: int, bits_per_tile: int = 3, minN: int = 21, maxN: int = 61) -> int:
    """Choose N so N*N*bits_per_tile >= header(7*8) + text*8 + crc(32). Prefer odd N like QR."""
    need = 7*8 + text_len*8 + 32
    # prefer odd N that fit quiet zone ratios already used in your code
    for N in range(minN, maxN+1, 2):
        if N*N*bits_per_tile >= need:
            return N
    return maxN

def choose_palette(name: str):
    palettes = {
        "basic": {
            "bg0": "#ffffff", "bg1": "#ffffff",
            "rope_under": "#000000", "rope_over": "#000000",
            "glow": "#333333", "dots": "#666666", "border": "#000000"
        },
        "indigo": {
            "bg0": "#0f0f23", "bg1": "#1a1a2e",
            "rope_under": "#3b82f6", "rope_over": "#e0e7ff",
            "glow": "#8ab4ff", "dots": "#f1f5f9", "border": "#1e293b"
        },
        "vermilion": {
            "bg0": "#1a0f0f", "bg1": "#2e1a1a",
            "rope_under": "#dc2626", "rope_over": "#fef2f2",
            "glow": "#fca5a5", "dots": "#f8fafc", "border": "#1f2937"
        },
        "jade": {
            "bg0": "#0f1a0f", "bg1": "#1a2e1a",
            "rope_under": "#059669", "rope_over": "#f0fdf4",
            "glow": "#6ee7b7", "dots": "#f8fafc", "border": "#0f172a"
        },
        "royal": {
            "bg0": "#0f0f1a", "bg1": "#1a1a2e",
            "rope_under": "#7c3aed", "rope_over": "#f3f4f6",
            "glow": "#a78bfa", "dots": "#f9fafb", "border": "#111827"
        },
        "classic": {
            "bg0": "#1a1a1a", "bg1": "#2d2d2d",
            "rope_under": "#f8f9fa", "rope_over": "#ffffff",
            "glow": "#e9ecef", "dots": "#ffffff", "border": "#343a40"
        }
    }
    return palettes.get(name, palettes["basic"])

def add_svg_defs(dwg, pal, glow_id="glow", rope_grad_id="ropegrad", paper_id="paper"):
    defs = dwg.defs

    # Rope linear gradient
    lg = svgwrite.gradients.LinearGradient(id=rope_grad_id, start=("0%", "0%"), end=("100%", "100%"))
    lg.add_stop_color(0, pal["rope_over"])
    lg.add_stop_color(1, pal["rope_under"])
    defs.add(lg)

    # Soft glow filter
    flt = dwg.filter(id=glow_id, x="-20%", y="-20%", width="140%", height="140%")
    flt.feGaussianBlur(in_="SourceGraphic", stdDeviation=2, result="blur")
    flt.feColorMatrix(
    type="matrix",
    values=[0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0.9,0],
    in_="blur",
    result="soft",
    )
    merge = flt.feMerge(["soft", "SourceGraphic"])
    defs.add(flt)

    # Subtle emboss filter
    emb = dwg.filter(id="emboss", x="-20%", y="-20%", width="140%", height="140%")
    emb.feGaussianBlur(in_="SourceAlpha", stdDeviation=0.7, result="alpha")
    emb.feSpecularLighting(in_="alpha", surfaceScale=2, specularConstant=0.6, specularExponent=15,
                           result="spec", lighting_color="#ffffff").fePointLight(x="0", y="-5000", z="2000")
    emb.feComposite(in_="spec", in2="SourceGraphic", operator="in", result="lit")
    emb.feMerge(["lit", "SourceGraphic"])
    defs.add(emb)

    # Paper texture via turbulence
    pap = dwg.filter(id=paper_id, x="0%", y="0%", width="100%", height="100%")
    pap.feTurbulence(type="fractalNoise", baseFrequency=0.8, numOctaves=2, seed=3, result="noise")
    pap.feColorMatrix(type="saturate", values=0.15, in_="noise", result="grain")
    pap.feBlend(in_="SourceGraphic", in2="grain", mode="multiply")
    defs.add(pap)

def mandala(dwg, center, r, strokes, stroke_width, stroke):
    g = dwg.g()
    cx, cy = center
    for k in range(strokes):
        a = 2*pi*k/strokes
        x1, y1 = cx + r*cos(a), cy + r*sin(a)
        x2, y2 = cx + (r*0.55)*cos(a+pi/6), cy + (r*0.55)*sin(a+pi/6)
        path_d = f"M {cx:.2f},{cy:.2f} Q {x2:.2f},{y2:.2f} {x1:.2f},{y1:.2f}"
        g.add(dwg.path(d=path_d, fill="none", stroke=stroke,
                       stroke_width=stroke_width, stroke_linecap="round"))
    return g

def _save_svg_or_png(dwg, out_path: str):
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".png":
        # For now, save as SVG and inform user
        svg_path = out_path.replace('.png', '.svg')
        dwg.saveas(svg_path)
        print(f"⚠️  PNG conversion not available. Saved as SVG: {svg_path}")
        print("💡 To convert SVG to PNG, you can:")
        print("   - Use online converters like svg2png.com")
        print("   - Install Inkscape and use: inkscape --export-type=png file.svg")
        print("   - Use browser: open SVG and save as PNG")
    else:
        dwg.saveas(out_path)

def draw_tile_rope(dwg, group, x, y, s, rot, curvature=0.62, grad="url(#ropegrad)",
                   under="#000", under_w=6.0, over_w=4.0, glow=None, simple_mode=False):
    """
    Draw kolam-style curves that flow around dot grid points.
    Enhanced for more organic, traditional kolam geometry.
    simple_mode: if True, uses basic black lines without complex effects
    """
    half = s/2.0
    cx, cy = x+half, y+half
    
    # Enhanced midpoint calculation for better flow around dots
    offset = s * 0.08  # slight offset to create space around dots
    mids = [
        (x + offset, cy), 
        (cx, y + offset), 
        (x + s - offset, cy), 
        (cx, y + s - offset)
    ]
    
    def rot_point(px, py, turns):
        dx, dy = px-cx, py-cy
        for _ in range(turns%4):
            dx, dy = -dy, dx
        return (cx+dx, cy+dy)

    L,T,R,B = [rot_point(px,py,rot) for (px,py) in mids]

    def kolam_bezier(p0, p1):
        # Enhanced bezier with kolam-style flowing curves
        k = curvature * 1.2  # More pronounced curves
        
        # Create flowing curves that respect dot spacing
        mid_x, mid_y = (p0[0] + p1[0])/2, (p0[1] + p1[1])/2
        
        # Control points that create the characteristic kolam flow
        c1_pull = k * 0.7
        c2_pull = k * 0.7
        
        c1 = (p0[0]*(1-c1_pull) + mid_x*c1_pull + (cy-mid_y)*0.3, 
              p0[1]*(1-c1_pull) + mid_y*c1_pull + (mid_x-cx)*0.3)
        c2 = (p1[0]*(1-c2_pull) + mid_x*c2_pull + (cy-mid_y)*0.3, 
              p1[1]*(1-c2_pull) + mid_y*c2_pull + (mid_x-cx)*0.3)
        
        return f"M {p0[0]:.2f},{p0[1]:.2f} C {c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p1[0]:.2f},{p1[1]:.2f}"

    # Draw the traditional two-arc pattern with enhanced styling
    for (p0,p1) in [(L,T),(R,B)]:
        d = kolam_bezier(p0,p1)
        
        if simple_mode:
            # Simple black line for basic palette - thicker for better contrast
            simple_path = dwg.path(d=d, fill="none", stroke="#000000",
                                  stroke_width=4.0, stroke_linecap="round", stroke_linejoin="round")
            group.add(simple_path)
        else:
            # Create layered effect for depth
            # Shadow/depth layer
            shadow_path = dwg.path(d=d, fill="none", stroke="#000000", stroke_opacity="0.3",
                                  stroke_width=under_w*1.3, stroke_linecap="round", stroke_linejoin="round")
            shadow_path.translate(1, 1)  # Slight offset for depth
            group.add(shadow_path)
            
            # Main underlay with gradient
            under_path = dwg.path(d=d, fill="none", stroke=grad,
                                  stroke_width=under_w, stroke_linecap="round", stroke_linejoin="round")
            under_path.update({"filter":"url(#emboss)"})
            group.add(under_path)
            
            # Highlight overlay
            over_path = dwg.path(d=d, fill="none", stroke="#ffffff",
                                 stroke_width=over_w, stroke_linecap="round", stroke_linejoin="round",
                                 opacity=0.65)
            if glow:
                over_path.update({"filter": glow})
            group.add(over_path)
            
            # Subtle inner highlight for traditional look
            inner_path = dwg.path(d=d, fill="none", stroke="#ffffff", stroke_opacity="0.8",
                                 stroke_width=max(1.0, over_w*0.3), stroke_linecap="round", stroke_linejoin="round")
            group.add(inner_path)

def _edges_for_rot(rot:int):
    # Edge names: 0=L,1=T,2=R,3=B
    # Base (rot=0): L↔T and R↔B
    if rot % 4 == 0:
        return [(0,1),(2,3)]
    if rot % 4 == 1:
        return [(1,2),(3,0)]  # T↔R, B↔L
    if rot % 4 == 2:
        return [(2,3),(0,1)]  # R↔B, L↔T (same pairs swapped)
    return [(3,0),(1,2)]      # B↔L, T↔R

def _draw_connector(dwg, g_art, x, y, tile_px, side, pal,
                    width_under, width_over, cusp=0.35):
    """
    Enhanced connector that creates flowing bridges between tiles.
    Creates the characteristic kolam weaving effect.
    """
    s = tile_px
    half = s/2.0
    cx, cy = x+half, y+half
    
    # Enhanced edge midpoints with offset for better flow
    offset = s * 0.08
    mids = [
        (x + offset, cy),
        (cx, y + offset), 
        (x + s - offset, cy),
        (cx, y + s - offset)
    ]
    px, py = mids[side]

    # Enhanced direction vectors for more organic flow
    normals = [(-1,0),(0,-1),(1,0),(0,1)]
    nx, ny = normals[side]
    tangents = [(0,-1),(1,0),(0,1),(-1,0)]
    tx, ty = tangents[side]

    # Enhanced control points for traditional kolam flow
    L = s*0.35                    # longer connector for better continuity
    C = L*cusp*1.4                # enhanced cusp for traditional look
    
    p0 = (px - nx*L/2, py - ny*L/2)
    p1 = (px + nx*L/2, py + ny*L/2)
    
    # More sophisticated control point calculation
    c0 = (px - nx*L*0.3 + tx*C + nx*L*0.1, py - ny*L*0.3 + ty*C + ny*L*0.1)
    c1 = (px + nx*L*0.3 + tx*C - nx*L*0.1, py + ny*L*0.3 + ty*C - ny*L*0.1)

    d = f"M {p0[0]:.2f},{p0[1]:.2f} C {c0[0]:.2f},{c0[1]:.2f} {c1[0]:.2f},{c1[1]:.2f} {p1[0]:.2f},{p1[1]:.2f}"
    
    # Enhanced layering for traditional look
    # Shadow
    shadow = dwg.path(d=d, fill="none", stroke="#000000", stroke_opacity="0.3",
                     stroke_width=width_under*1.2, stroke_linecap="round", stroke_linejoin="round")
    shadow.translate(0.5, 0.5)
    g_art.add(shadow)
    
    # Main gradient layer
    under = dwg.path(d=d, fill="none", stroke="url(#ropegrad)",
                     stroke_width=width_under, stroke_linecap="round", stroke_linejoin="round")
    under.update({"filter":"url(#emboss)"})
    g_art.add(under)
    
    # Highlight
    over = dwg.path(d=d, fill="none", stroke="#ffffff",
                    stroke_width=width_over, stroke_linecap="round", stroke_linejoin="round", opacity=0.7)
    over.update({"filter":"url(#glow)"})
    g_art.add(over)

def add_background(dwg, size, pal):
    # radial vignette
    bg = dwg.add(dwg.g())
    bg.add(dwg.rect(insert=(0,0), size=(size,size), fill=pal["bg0"]))
    rad = svgwrite.gradients.RadialGradient(id="vign", cx="50%", cy="50%", r="65%")
    rad.add_stop_color(0, pal["bg1"])
    rad.add_stop_color(1, pal["bg0"])
    dwg.defs.add(rad)
    bg.add(dwg.rect(insert=(0,0), size=(size,size), fill="url(#vign)", **{"filter":"url(#paper)"}))
    return bg

def add_pulli_lattice(dwg, canvas, margin, size, spacing, color, opacity=0.25, r=1.4):
    """Add the foundational dot grid (pulli) that kolam patterns weave around"""
    g = dwg.g(opacity=opacity)
    x0 = margin; x1 = size - margin
    y0 = margin; y1 = size - margin
    # snap spacing to fit nicely
    xs = int((x1-x0)/spacing)+1
    ys = int((y1-y0)/spacing)+1
    
    # Create prominent dots with traditional kolam styling
    for i in range(xs):
        for j in range(ys):
            x = x0 + i*spacing
            y = y0 + j*spacing
            # Main dot (larger and more visible)
            g.add(dwg.circle(center=(x,y), r=r*1.8, fill=color, opacity=0.9))
            # Subtle highlight ring
            g.add(dwg.circle(center=(x,y), r=r*1.2, fill="none", stroke=color, 
                            stroke_width=0.5, opacity=0.6))
    canvas.add(g)

def draw_tile_clean(dwg, g_art, g_decode, x, y, s, rot, dotbit,
                    stroke_decode="black", decode_w=1.7, curvature=0.62):
    """Draw clean style: single continuous line with proper alignment."""
    half = s/2.0
    cx, cy = x+half, y+half
    
    # Calculate precise connection points that align with grid
    # These points are exactly at tile edges for perfect alignment
    edge_offset = 0  # No offset - connect directly at edges
    mids = [
        (x + edge_offset, cy),         # Left edge
        (cx, y + edge_offset),         # Top edge  
        (x + s - edge_offset, cy),     # Right edge
        (cx, y + s - edge_offset)      # Bottom edge
    ]
    
    def rot_point(px, py, turns):
        dx, dy = px-cx, py-cy
        for _ in range(turns%4):
            dx, dy = -dy, dx
        return (cx+dx, cy+dy)

    L,T,R,B = [rot_point(px,py,rot) for (px,py) in mids]

    # Draw single continuous flowing line with perfect edge alignment
    def clean_curve(p0, p1):
        # Create a smooth curve that flows from edge to edge
        # Control points create gentle curves without gaps
        mid_x, mid_y = (p0[0] + p1[0])/2, (p0[1] + p1[1])/2
        
        # Subtle control points for smooth flow
        ctrl_offset = s * 0.25  # Reduced for cleaner lines
        
        # Calculate control points for smooth S-curve
        c1x = p0[0] + (mid_x - p0[0]) * 0.5 + (cy - mid_y) * 0.2
        c1y = p0[1] + (mid_y - p0[1]) * 0.5 + (mid_x - cx) * 0.2
        c2x = p1[0] + (mid_x - p1[0]) * 0.5 + (cy - mid_y) * 0.2  
        c2y = p1[1] + (mid_y - p1[1]) * 0.5 + (mid_x - cx) * 0.2
        
        d = f"M {p0[0]:.2f},{p0[1]:.2f} C {c1x:.2f},{c1y:.2f} {c2x:.2f},{c2y:.2f} {p1[0]:.2f},{p1[1]:.2f}"
        
        # Single clean line - thicker for better visibility
        path = dwg.path(d=d, fill="none", stroke="#000000",
                       stroke_width=2.5, stroke_linecap="round", stroke_linejoin="round")
        g_art.add(path)

    # Draw the two main curves that define the kolam pattern
    clean_curve(L, T)  # Left to Top
    clean_curve(R, B)  # Right to Bottom

    # Decodable layer - thin arcs for QR functionality
    r = 0.95*half
    def arc_cmd(a1, a2):
        def pt(a):
            rad = a*pi/180.0
            return (cx + r*cos(rad), cy + r*sin(rad))
        p0 = pt(a1); p1 = pt(a2)
        return f"M {p0[0]:.2f},{p0[1]:.2f} A {r:.2f},{r:.2f} 0 0 1 {p1[0]:.2f},{p1[1]:.2f}"
    
    # Rotation mapping for decode layer
    if rot == 0:
        arcs = [(180,270),(0,90)]
    elif rot == 1:
        arcs = [(270,360),(90,180)]
    elif rot == 2:
        arcs = [(0,90),(180,270)]
    else:
        arcs = [(90,180),(270,360)]
    
    for a1,a2 in arcs:
        path = dwg.path(d=arc_cmd(a1,a2), fill="none", stroke=stroke_decode,
                        stroke_width=decode_w, stroke_linecap="round")
        g_decode.add(path)
    
    if dotbit == 1:
        # Decode layer: small black dot for QR functionality
        g_decode.add(dwg.circle(center=(cx,cy), r=s*0.06, fill=stroke_decode))
        
        # Art layer: Simple dot with minimal decoration
        g_art.add(dwg.circle(center=(cx,cy), r=s*0.04, fill="#000000"))

def draw_tile_fancy(dwg, g_art, g_decode, x, y, s, rot, dotbit,
                    stroke_decode="black", decode_w=1.7, curvature=0.62, palette="basic"):
    """Draw both: pretty rope layer + thin decodable arcs + center dot on decode layer."""
    # Check if using simple mode
    simple_mode = (palette == "basic")
    
    # pretty rope
    draw_tile_rope(dwg, g_art, x, y, s, rot,
                   curvature=curvature, grad="url(#ropegrad)",
                   under_w=max(4.0, s*0.11), over_w=max(2.5, s*0.07), glow="url(#glow)",
                   simple_mode=simple_mode)

    # original decodable arcs (thin) — recreate with same geometry using quarter-ellipses
    half = s/2; r = 0.95*half; cx, cy = x+half, y+half
    def arc_cmd(a1, a2):
        # convert degrees to end point on circle
        def pt(a):
            rad = a*pi/180.0
            return (cx + r*cos(rad), cy + r*sin(rad))
        p0 = pt(a1); p1 = pt(a2)
        return f"M {p0[0]:.2f},{p0[1]:.2f} A {r:.2f},{r:.2f} 0 0 1 {p1[0]:.2f},{p1[1]:.2f}"
    # rotation mapping
    if rot == 0:
        arcs = [(180,270),(0,90)]
    elif rot == 1:
        arcs = [(270,360),(90,180)]
    elif rot == 2:
        arcs = [(0,90),(180,270)]
    else:
        arcs = [(90,180),(270,360)]
    for a1,a2 in arcs:
        path = dwg.path(d=arc_cmd(a1,a2), fill="none", stroke=stroke_decode,
                        stroke_width=decode_w, stroke_linecap="round")
        g_decode.add(path)
    if dotbit == 1:
        # decode layer: keep small black dot
        g_decode.add(dwg.circle(center=(cx,cy), r=s*0.06, fill=stroke_decode))
        
        # art layer: Traditional kolam lotus/star pattern around dot
        pet = dwg.g(opacity=0.8)
        
        # Create multi-layered petal design
        # Outer petals (8-fold)
        rr_outer = s*0.22
        for k in range(8):
            ang = k*pi/4
            x1, y1 = cx + rr_outer*cos(ang), cy + rr_outer*sin(ang)
            x2, y2 = cx + (rr_outer*0.6)*cos(ang+pi/8), cy + (rr_outer*0.6)*sin(ang+pi/8)
            x3, y3 = cx + (rr_outer*0.6)*cos(ang-pi/8), cy + (rr_outer*0.6)*sin(ang-pi/8)
            
            # Create traditional petal shape
            dstar = f"M {cx:.2f},{cy:.2f} Q {x2:.2f},{y2:.2f} {x1:.2f},{y1:.2f} Q {x3:.2f},{y3:.2f} {cx:.2f},{cy:.2f}"
            pet.add(dwg.path(d=dstar, fill="none", stroke="#000000", 
                            stroke_width=max(0.6, s*0.015), stroke_linecap="round"))
        
        # Inner petals (4-fold)
        rr_inner = s*0.12
        for k in range(4):
            ang = k*pi/2 + pi/4  # offset by 45 degrees
            x1, y1 = cx + rr_inner*cos(ang), cy + rr_inner*sin(ang)
            xq, yq = cx + (rr_inner*0.5)*cos(ang+pi/6), cy + (rr_inner*0.5)*sin(ang+pi/6)
            dstar = f"M {cx:.2f},{cy:.2f} Q {xq:.2f},{yq:.2f} {x1:.2f},{y1:.2f}"
            pet.add(dwg.path(d=dstar, fill="none", stroke="#000000", 
                            stroke_width=max(0.4, s*0.01), stroke_linecap="round"))
        
        # Central ring
        pet.add(dwg.circle(center=(cx,cy), r=s*0.08, fill="none", stroke="#000000", 
                          stroke_width=max(0.5, s*0.012), opacity=0.6))
        
        g_art.add(pet)

def draw_tile_fancy(dwg, g_art, g_decode, x, y, s, rot, dotbit,
                    stroke_decode="black", decode_w=1.7, curvature=0.62, palette="basic"):
    """Draw both: pretty rope layer + thin decodable arcs + center dot on decode layer."""
    # Check if using simple mode
    simple_mode = (palette == "basic")
    
    # pretty rope
    draw_tile_rope(dwg, g_art, x, y, s, rot,
                   curvature=curvature, grad="url(#ropegrad)",
                   under_w=max(4.0, s*0.11), over_w=max(2.5, s*0.07), glow="url(#glow)",
                   simple_mode=simple_mode)

    # original decodable arcs (thin) — recreate with same geometry using quarter-ellipses
    half = s/2; r = 0.95*half; cx, cy = x+half, y+half
    def arc_cmd(a1, a2):
        # convert degrees to end point on circle
        def pt(a):
            rad = a*pi/180.0
            return (cx + r*cos(rad), cy + r*sin(rad))
        p0 = pt(a1); p1 = pt(a2)
        return f"M {p0[0]:.2f},{p0[1]:.2f} A {r:.2f},{r:.2f} 0 0 1 {p1[0]:.2f},{p1[1]:.2f}"
    # rotation mapping
    if rot == 0:
        arcs = [(180,270),(0,90)]
    elif rot == 1:
        arcs = [(270,360),(90,180)]
    elif rot == 2:
        arcs = [(0,90),(180,270)]
    else:
        arcs = [(90,180),(270,360)]
    for a1,a2 in arcs:
        path = dwg.path(d=arc_cmd(a1,a2), fill="none", stroke=stroke_decode,
                        stroke_width=decode_w, stroke_linecap="round")
        g_decode.add(path)
    if dotbit == 1:
        # decode layer: keep small black dot
        g_decode.add(dwg.circle(center=(cx,cy), r=s*0.06, fill=stroke_decode))
        
        # art layer: Traditional kolam lotus/star pattern around dot
        pet = dwg.g(opacity=0.8)
        
        # Create multi-layered petal design
        # Outer petals (8-fold)
        rr_outer = s*0.22
        for k in range(8):
            ang = k*pi/4
            x1, y1 = cx + rr_outer*cos(ang), cy + rr_outer*sin(ang)
            x2, y2 = cx + (rr_outer*0.6)*cos(ang+pi/8), cy + (rr_outer*0.6)*sin(ang+pi/8)
            x3, y3 = cx + (rr_outer*0.6)*cos(ang-pi/8), cy + (rr_outer*0.6)*sin(ang-pi/8)
            
            # Create traditional petal shape
            dstar = f"M {cx:.2f},{cy:.2f} Q {x2:.2f},{y2:.2f} {x1:.2f},{y1:.2f} Q {x3:.2f},{y3:.2f} {cx:.2f},{cy:.2f}"
            pet.add(dwg.path(d=dstar, fill="none", stroke="#000000", 
                            stroke_width=max(0.6, s*0.015), stroke_linecap="round"))
        
        # Inner petals (4-fold)
        rr_inner = s*0.12
        for k in range(4):
            ang = k*pi/2 + pi/4  # offset by 45 degrees
            x1, y1 = cx + rr_inner*cos(ang), cy + rr_inner*sin(ang)
            xq, yq = cx + (rr_inner*0.5)*cos(ang+pi/6), cy + (rr_inner*0.5)*sin(ang+pi/6)
            dstar = f"M {cx:.2f},{cy:.2f} Q {xq:.2f},{yq:.2f} {x1:.2f},{y1:.2f}"
            pet.add(dwg.path(d=dstar, fill="none", stroke="#000000", 
                            stroke_width=max(0.4, s*0.01), stroke_linecap="round"))
        
        # Central ring
        pet.add(dwg.circle(center=(cx,cy), r=s*0.08, fill="none", stroke="#000000", 
                          stroke_width=max(0.5, s*0.012), opacity=0.6))
        
        g_art.add(pet)

def render_svg_clean(grid_bits: List[int], N: int, tile_bits=3,
                     out_svg="kolam.svg", tile_px=42, margin_px=180, stroke_px=3.0,
                     dot_opacity=None):
    """Clean renderer: sophisticated kolam patterns with just simple lines and dots."""
    total_tiles = N*N
    bits = grid_bits[: total_tiles*tile_bits] + [0]*max(0, total_tiles*tile_bits - len(grid_bits))

    size = margin_px*2 + N*tile_px
    dwg = svgwrite.Drawing(out_svg, size=(size, size))
    
    # Simple white background
    bg = dwg.add(dwg.g())
    bg.add(dwg.rect(insert=(0,0), size=(size,size), fill="#ffffff"))

    # Simple black border
    border = dwg.add(dwg.g())
    border.add(dwg.rect(insert=(2,2), size=(size-4, size-4),
                        stroke="#000000", fill="none", stroke_width=4))

    # Simple finder circles (just black circles)
    finder_r = margin_px*0.22
    for c in [(margin_px*0.6, margin_px*0.6),
              (size - margin_px*0.6, margin_px*0.6),
              (margin_px*0.6, size - margin_px*0.6)]:
        border.add(dwg.circle(center=c, r=finder_r, fill="#000000"))

    # Simple dot lattice
    dot_op = dot_opacity if dot_opacity is not None else 0.4
    spacing = max(8, tile_px*0.4)
    
    g_dots = dwg.g(opacity=dot_op)
    x0 = margin_px; x1 = size - margin_px
    y0 = margin_px; y1 = size - margin_px
    xs = int((x1-x0)/spacing)+1
    ys = int((y1-y0)/spacing)+1
    
    for i in range(xs):
        for j in range(ys):
            x = x0 + i*spacing
            y = y0 + j*spacing
            # Simple small dots
            g_dots.add(dwg.circle(center=(x,y), r=2, fill="#666666"))
    border.add(g_dots)

    # layers
    g_art = dwg.add(dwg.g())       # kolam lines
    g_decode = dwg.add(dwg.g())    # thin arcs that decoder will read

    # tiles
    k = 0
    for r in range(N):
        for c in range(N):
            b0, b1, b2 = bits[k], bits[k+1], bits[k+2]
            k += 3
            rot = (b0<<1) | b1
            x = margin_px + c*tile_px
            y = margin_px + r*tile_px
            draw_tile_clean(dwg, g_art, g_decode, x, y, tile_px, rot, b2, curvature=0.68)

    # ---- second pass: simple connectors ----
    # cache rotations to avoid recompute
    rots = np.zeros((N,N), dtype=int)
    k = 0
    for r in range(N):
        for c in range(N):
            b0, b1, b2 = bits[k], bits[k+1], bits[k+2]
            rots[r,c] = (b0<<1) | b1
            k += 3

    # Continuous flowing connectors
    def draw_clean_connector(dwg, g_art, x, y, tile_px, side):
        s = tile_px
        half = s/2.0
        cx, cy = x+half, y+half
        
        # No offset - connect directly at tile edges for perfect alignment
        mids = [
            (x, cy),           # Left edge (x=0)
            (cx, y),           # Top edge (y=0)  
            (x + s, cy),       # Right edge (x=s)
            (cx, y + s)        # Bottom edge (y=s)
        ]
        px, py = mids[side]

        normals = [(-1,0),(0,-1),(1,0),(0,1)]
        nx, ny = normals[side]
        
        # Longer connector for better continuity
        L = s*0.5  # Extended length
        p0 = (px - nx*L/2, py - ny*L/2)
        p1 = (px + nx*L/2, py + ny*L/2)
        
        # Smooth curved connector instead of straight line
        # Create subtle curve for natural flow
        tangents = [(0,-1),(1,0),(0,1),(-1,0)]
        tx, ty = tangents[side]
        
        # Control points for gentle S-curve
        cusp = L*0.15  # Subtle curvature
        c0 = (px - nx*L*0.3 + tx*cusp, py - ny*L*0.3 + ty*cusp)
        c1 = (px + nx*L*0.3 + tx*cusp, py + ny*L*0.3 + ty*cusp)
        
        d = f"M {p0[0]:.2f},{p0[1]:.2f} C {c0[0]:.2f},{c0[1]:.2f} {c1[0]:.2f},{c1[1]:.2f} {p1[0]:.2f},{p1[1]:.2f}"
        
        # Single flowing line matching tile stroke width
        g_art.add(dwg.path(d=d, fill="none", stroke="#000000", 
                          stroke_width=2.5, stroke_linecap="round", stroke_linejoin="round"))

    # Horizontal neighbours
    for r in range(N):
        for c in range(N-1):
            rotA = rots[r,c]; rotB = rots[r,c+1]
            edgesA = _edges_for_rot(rotA)
            edgesB = _edges_for_rot(rotB)
            touchA = any(2 in p for p in edgesA)
            touchB = any(0 in p for p in edgesB)
            if touchA and touchB:
                x = margin_px + c*tile_px
                y = margin_px + r*tile_px
                draw_clean_connector(dwg, g_art, x, y, tile_px, side=2)

    # Vertical neighbours
    for r in range(N-1):
        for c in range(N):
            rotA = rots[r,c]; rotB = rots[r+1,c]
            edgesA = _edges_for_rot(rotA)
            edgesB = _edges_for_rot(rotB)
            touchA = any(3 in p for p in edgesA)
            touchB = any(1 in p for p in edgesB)
            if touchA and touchB:
                x = margin_px + c*tile_px
                y = margin_px + r*tile_px
                draw_clean_connector(dwg, g_art, x, y, tile_px, side=3)

    dwg.save()
    return out_svg

def render_svg_fancy(grid_bits: List[int], N: int, tile_bits=3,
                     out_svg="kolam.svg", tile_px=42, margin_px=180, stroke_px=3.0,
                     palette="indigo", fractal_depth=0, dot_opacity=None):
    """Pretty renderer that keeps a thin, decodable layer on top."""
    total_tiles = N*N
    bits = grid_bits[: total_tiles*tile_bits] + [0]*max(0, total_tiles*tile_bits - len(grid_bits))

    size = margin_px*2 + N*tile_px
    dwg = svgwrite.Drawing(out_svg, size=(size, size))
    pal = choose_palette(palette)
    add_svg_defs(dwg, pal)

    # Background + paper
    add_background(dwg, size, pal)

    # border
    border = dwg.add(dwg.g())
    border.add(dwg.rect(insert=(2,2), size=(size-4, size-4),
                        stroke=pal["border"], fill="none", stroke_width=4))

    # finders (outer circles unchanged but tinted)
    finder_r = margin_px*0.22
    for c in [(margin_px*0.6, margin_px*0.6),
              (size - margin_px*0.6, margin_px*0.6),
              (margin_px*0.6, size - margin_px*0.6)]:
        border.add(dwg.circle(center=c, r=finder_r, fill=pal["rope_under"], opacity=0.9))
        border.add(mandala(dwg, c, finder_r*0.78, strokes=18, stroke_width=1.2, stroke=pal["rope_over"]))

    # pulli lattice - make it more prominent for traditional look
    # Determine dot opacity based on palette or user override
    if dot_opacity is not None:
        dot_op = dot_opacity
    elif palette == "basic":
        dot_op = 0.6  # Grey dots with moderate opacity for basic palette
    else:
        dot_op = 0.85  # Default for other palettes
        
    add_pulli_lattice(dwg, border, margin_px, size, spacing=max(8, tile_px*0.4),
                      color=pal["dots"], opacity=dot_op, r=max(3.0, tile_px*0.05))

    # layers
    g_art = dwg.add(dwg.g())       # pretty rope underlay
    g_decode = dwg.add(dwg.g())    # thin arcs that decoder will read

    # tiles
    k = 0
    for r in range(N):
        for c in range(N):
            b0, b1, b2 = bits[k], bits[k+1], bits[k+2]
            k += 3
            rot = (b0<<1) | b1
            x = margin_px + c*tile_px
            y = margin_px + r*tile_px
            draw_tile_fancy(dwg, g_art, g_decode, x, y, tile_px, rot, b2, curvature=0.68, palette=palette)

            # optional micro-fractal: faint mini-kolam inside (does not affect decoding)
            if fractal_depth > 0 and tile_px >= 26:
                sub = dwg.g(opacity=0.18)
                size_sub = tile_px*0.6
                simple_mode = (palette == "basic")
                draw_tile_rope(dwg, sub, x+(tile_px-size_sub)/2, y+(tile_px-size_sub)/2,
                               size_sub, (rot+1)%4, curvature=0.65,
                               grad="url(#ropegrad)", under_w=max(2.0, tile_px*0.06),
                               over_w=max(1.2, tile_px*0.035), simple_mode=simple_mode)
                g_art.add(sub)

    # ---- second pass: stitch neighbours with kolam cusps ----
    # cache rotations to avoid recompute
    rots = np.zeros((N,N), dtype=int)
    dots = np.zeros((N,N), dtype=int)
    k = 0
    for r in range(N):
        for c in range(N):
            b0, b1, b2 = bits[k], bits[k+1], bits[k+2]
            rots[r,c] = (b0<<1) | b1
            dots[r,c] = b2
            k += 3

    wu = max(4.0, tile_px*0.11)  # under width (same as tile rope)
    wo = max(2.5, tile_px*0.07)  # over width

    # Horizontal neighbours
    for r in range(N):
        for c in range(N-1):
            rotA = rots[r,c]; rotB = rots[r,c+1]
            edgesA = _edges_for_rot(rotA)
            edgesB = _edges_for_rot(rotB)
            # they share A:Right(2) <-> B:Left(0)
            touchA = any(2 in p for p in edgesA)
            touchB = any(0 in p for p in edgesB)
            if touchA and touchB:
                x = margin_px + c*tile_px
                y = margin_px + r*tile_px
                # draw centered on the edge; one call per pair (draw once on left tile)
                _draw_connector(dwg, g_art, x, y, tile_px, side=2, pal=pal,
                                width_under=wu, width_over=wo, cusp=0.40)

    # Vertical neighbours
    for r in range(N-1):
        for c in range(N):
            rotA = rots[r,c]; rotB = rots[r+1,c]
            edgesA = _edges_for_rot(rotA)
            edgesB = _edges_for_rot(rotB)
            # they share A:Bottom(3) <-> B:Top(1)
            touchA = any(3 in p for p in edgesA)
            touchB = any(1 in p for p in edgesB)
            if touchA and touchB:
                x = margin_px + c*tile_px
                y = margin_px + r*tile_px
                _draw_connector(dwg, g_art, x, y, tile_px, side=3, pal=pal,
                                width_under=wu, width_over=wo, cusp=0.40)

    _save_svg_or_png(dwg, out_svg)
    return out_svg








if __name__ == "__main__":
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    args.func(args)

