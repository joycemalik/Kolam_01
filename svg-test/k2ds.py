import argparse, base64, binascii, json, math, os, sys
from typing import Tuple, List
import numpy as np
import cv2
import svgwrite
from xml.etree import ElementTree as ET
from PIL import Image
import hashlib
from math import cos, sin, pi

# =========================
# Robust SVG Support Detection
# =========================
CAIROSVG_AVAILABLE = False
SVGLIB_AVAILABLE = False
SVG2PNG_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
    print("✅ CairoSVG available")
except (ImportError, OSError) as e:
    CAIROSVG_AVAILABLE = False
    print(f"⚠️  CairoSVG not available: {e}")

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    SVGLIB_AVAILABLE = True
    print("✅ SVGLib available")
except ImportError as e:
    SVGLIB_AVAILABLE = False
    print(f"⚠️  SVGLib not available: {e}")

# Alternative: Use PIL for basic SVG loading (limited but works)
try:
    from PIL import Image as PILImage
    SVG2PNG_AVAILABLE = True
except ImportError:
    SVG2PNG_AVAILABLE = False

def get_supported_formats() -> List[str]:
    """Return list of supported image formats"""
    basic_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif']
    svg_formats = ['.svg']
    
    # Check if SVG support is available
    svg_available = CAIROSVG_AVAILABLE or SVGLIB_AVAILABLE or SVG2PNG_AVAILABLE
    
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
        print("  • pip install Pillow (basic SVG support)")
    else:
        if CAIROSVG_AVAILABLE:
            svg_backend = "cairosvg"
        elif SVGLIB_AVAILABLE:
            svg_backend = "svglib+reportlab" 
        else:
            svg_backend = "Pillow (basic)"
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
def pack_header(n_tiles: int, text_len: int, version: int = 1) -> bytes:
    """
    Header: 1 byte version | 2 bytes N (grid) | 4 bytes text_len | 4 bytes CRC32(payload)
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
# Robust SVG Loading
# =========================
def load_image_universal(img_path: str) -> np.ndarray:
    """
    Robust image loader with multiple fallbacks for SVG and other formats
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    file_ext = os.path.splitext(img_path)[1].lower()
    
    try:
        if file_ext == '.svg':
            print(f"Converting SVG to image...")
            
            # Method 1: Try cairosvg first
            if CAIROSVG_AVAILABLE:
                try:
                    png_data = cairosvg.svg2png(url=img_path)
                    img_array = np.frombuffer(png_data, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    print("✅ SVG loaded via cairosvg")
                    return _convert_to_grayscale(img)
                except Exception as svg_err:
                    print(f"❌ cairosvg failed: {svg_err}")
            
            # Method 2: Try svglib + reportlab
            if SVGLIB_AVAILABLE:
                try:
                    drawing = svg2rlg(img_path)
                    img_pil = renderPM.drawToPIL(drawing)
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    print("✅ SVG loaded via svglib")
                    return _convert_to_grayscale(img)
                except Exception as pil_err:
                    print(f"❌ svglib failed: {pil_err}")
            
            # Method 3: Try PIL (basic SVG support)
            if SVG2PNG_AVAILABLE:
                try:
                    img_pil = PILImage.open(img_path)
                    if hasattr(img_pil, '_is_animated') and img_pil._is_animated:
                        img_pil.seek(0)  # Use first frame for animated SVGs
                    img_pil = img_pil.convert('RGB')
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    print("✅ SVG loaded via PIL")
                    return _convert_to_grayscale(img)
                except Exception as pil_err:
                    print(f"❌ PIL failed: {pil_err}")
            
            # Method 4: Last resort - use browser or external tool
            raise RuntimeError(
                "SVG support requires one of:\n"
                "• pip install cairosvg\n" 
                "• pip install svglib reportlab\n"
                "• pip install Pillow\n\n"
                "Alternatively, convert SVG to PNG manually first."
            )
        
        else:
            # Handle standard image formats
            return _load_standard_image(img_path, file_ext)
            
    except Exception as e:
        raise RuntimeError(f"Error loading image {img_path}: {e}")

def _load_standard_image(img_path: str, file_ext: str) -> np.ndarray:
    """Load standard image formats with fallbacks"""
    # Try OpenCV first
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is not None:
        print(f"✅ Loaded {file_ext} via OpenCV")
        return _convert_to_grayscale(img)
    
    # Fallback to PIL
    try:
        pil_img = Image.open(img_path)
        # Handle different modes
        if pil_img.mode == 'RGBA':
            # Convert RGBA to RGB
            background = Image.new('RGB', pil_img.size, (255, 255, 255))
            background.paste(pil_img, mask=pil_img.split()[-1])
            pil_img = background
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        print(f"✅ Loaded {file_ext} via PIL")
        return _convert_to_grayscale(img)
    except Exception as pil_err:
        print(f"❌ PIL failed: {pil_err}")
    
    raise RuntimeError(f"Unsupported image format or corrupted file: {file_ext}")

def _convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    print(f"Image dimensions: {gray.shape}")
    return gray

# =========================
# Tile drawing (SVG) - Keep your existing functions
# =========================
def draw_tile(dwg, group, x, y, s, rot, dotbit, stroke="black", stroke_width=2.0):
    """Kolam-like tile: two quarter arcs connecting midpoints."""
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
        d_string = (f"M {p0[0]:.3f},{p0[1]:.3f} "
                f"A {r:.3f},{r:.3f} 0 0 1 {p1[0]:.3f},{p1[1]:.3f}")
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
# Decoder helpers - Keep your existing functions
# =========================
def find_border_alternative(gray: np.ndarray):
    """Alternative border detection for images without clear rectangular borders"""
    h, w = gray.shape
    
    # Try to find the content area by detecting the kolam pattern itself
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
    # Your existing implementation
    h, w = gray.shape
    min_area = (h * w) * 0.1
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    
    _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    _, th3 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    best = None
    best_area = 0
    best_th = None
    
    for i, th in enumerate([th1, th2, th3]):
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            peri = cv2.arcLength(cnt, True)
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
    
    if best is None:
        print("No rectangular border found, trying content-based detection...")
        best = find_border_alternative(gray)
        best_th = th1
    
    return best, best_th if best_th is not None else th1

def order_corners(pts: np.ndarray) -> np.ndarray:
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

def build_tile_templates(px: int, stroke: int=3) -> List[np.ndarray]:
    # Your existing implementation
    s = px
    
    def tile_image(rotation):
        im = np.zeros((s, s), np.uint8)
        half = s // 2
        r = int(0.95 * s/2)
        
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
    """Return (rot, dotbit)."""
    h, w = tile_roi.shape[:2]
    th = cv2.threshold(tile_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    T = templates[0].shape[0]
    roi = cv2.resize(th, (T, T), interpolation=cv2.INTER_AREA)

    scores = []
    for t in templates:
        s = (t.astype(np.float32) / 255.0)
        r = (roi.astype(np.float32) / 255.0)
        num = np.sum(s * r)
        den = math.sqrt(np.sum(s*s) * np.sum(r*r) + 1e-6)
        scores.append(num / den)
    rot = int(np.argmax(scores))

    mask = np.zeros((T, T), np.uint8)
    cv2.circle(mask, (T//2, T//2), int(0.08*T), 255, -1)
    dot_pixels = np.sum(roi[mask==255]) / 255.0
    dotbit = 1 if dot_pixels > (0.35 * np.sum(mask==255)) else 0
    return rot, dotbit

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
        print(f"Image shape: {gray.shape}")
        print(f"Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")
        for i, cnt in enumerate(contours[:5]):
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(f"  Contour {i}: area={area:.0f}, vertices={len(approx)}")
        
        raise RuntimeError("Could not find outer border. Try:\n"
                          "1. Ensure the image has a clear black rectangular border\n"
                          "2. Increase image contrast\n"
                          "3. Use --debug flag to see detection images\n"
                          "4. Check that the border is at least 4 pixels thick")

    OUT = 1600
    warped = warp_to_square(gray, quad, OUT)
    
    if debug:
        cv2.imwrite('debug_warped.png', warped)
        print(f"Saved debug_warped.png - Perspective corrected image")

    margin_px = int(OUT * 0.12)
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
            
        header_bits = bits[:8*7]
        header = bits_to_bytes(header_bits)
        try:
            version, n_tiles, text_len = unpack_header(header)
        except Exception as e:
            last_err = e
            continue
            
        if n_tiles != N*N or text_len < 0 or text_len > 200000:
            last_err = ValueError("Header sanity failed")
            continue

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
    N = args.grid if args.grid else auto_grid_for_text(len(text.encode("utf-8")), bits_per_tile=3)

    bits = bytes_to_bits(pack_header(N*N, len(text.encode('utf-8')))) \
           + bytes_to_bits(text.encode('utf-8')) \
           + bytes_to_bits(crc32(text.encode('utf-8')).to_bytes(4, 'big'))

    # Use basic renderer for maximum compatibility
    svg_path = render_svg(bits, N=N, tile_bits=3,
                          out_svg=args.out, tile_px=args.tile_px,
                          margin_px=args.margin_px, stroke_px=args.stroke_px)

    print(f"✅ Saved: {svg_path}")
    print(f"📊 Grid: {N}x{N}, capacity bits: {N*N*3}, used bits: {len(bits)}")

def decode_cmd(args):
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
    e.add_argument("--stroke-px", type=float, default=3.0, help="stroke width")
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
    for N in range(minN, maxN+1, 2):
        if N*N*bits_per_tile >= need:
            return N
    return maxN

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    # Print dependency status at startup
    print("🔍 Checking dependencies...")
    print_format_info()
    print("-" * 50)
    
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if "CUDA" in str(e) or "GPU" in str(e):
            print("💡 Try setting environment variable: CUDA_VISIBLE_DEVICES=-1")
        sys.exit(1)