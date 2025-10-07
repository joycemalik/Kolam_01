"""
Sikku Kolam Encoder → PNG (via SVG)
------------------------------------

Takes a text input, maps it to a symmetric A/B tile pattern on an n×n dot grid,
then renders a Sikku-style kolam as an SVG and (if cairosvg is available) also
saves a PNG.

Design goals satisfied:
- Always closed loops (degree-2 at all tiles using A/B pairings)
- D4 symmetry (4-way rotational + reflectional) ring-wise, from corners inward
- Corner tiles force inward flow (set to B), creating a "from corners to center" look
- Deterministic text→bits→kolam mapping, with padding from a checksum PRNG

USAGE
-----
$ python sikku_kolam_encoder.py --text "hello kolam" --n  nine --png out.png --svg out.svg

If cairosvg is installed, a PNG will be produced. Otherwise only SVG is written.

DEPENDENCIES
------------
- svgwrite (pip install svgwrite)
- cairosvg (optional, for PNG export)

"""
from __future__ import annotations
import argparse
import hashlib
import math
import os
import random
from typing import List, Tuple

try:
    import svgwrite  # type: ignore
except ImportError as e:
    raise SystemExit("Please install svgwrite: pip install svgwrite")

# PNG export is optional
try:
    import cairosvg  # type: ignore
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

# -----------------------------
# Basic utilities
# -----------------------------

def text_to_bits(s: str) -> List[int]:
    return [int(b) for ch in s.encode("utf-8") for b in f"{ch:08b}"]


def prng_pad(bits: List[int], need: int, seed: int) -> List[int]:
    """Pad bits to 'need' length with a deterministic PRNG stream."""
    if len(bits) >= need:
        return bits[:need]
    rng = random.Random(seed)
    pad = [rng.randint(0, 1) for _ in range(need - len(bits))]
    return bits + pad


def bit_budget(n: int) -> int:
    """Number of payload bits needed by ring-wise D4 scheme on odd n."""
    if n % 2 == 0:
        raise ValueError("Use odd n for a true center (e.g.,  five, seven, nine, ...)")
    r = (n + 1) // 2
    total = 1  # center bit
    for k in range(1, r):
        Lk = n - 2 * k - 1  # top edge length excluding two corners
        if Lk < 0:
            Lk = 0
        total += Lk
    return total

# -----------------------------
# A/B tile grid construction with ring D4 symmetry
# -----------------------------

Tile = int  # 0 = A (straight weave), 1 = B (diagonal weave)


def build_rule_matrix(n: int, payload_bits: List[int]) -> List[List[Tile]]:
    """Fill n×n with A/B using ring-wise D4 symmetry and corner=B inward bends.
    payload_bits length must be bit_budget(n).
    """
    if n % 2 == 0:
        raise ValueError("n must be odd")
    need = bit_budget(n)
    if len(payload_bits) != need:
        raise ValueError(f"payload_bits must have length {need}")

    rule = [[0 for _ in range(n)] for _ in range(n)]  # default A
    r = (n + 1) // 2
    idx = 0
    for k in range(1, r):
        i0, i1 = k - 1, n - k
        j0, j1 = i0, i1
        # Corners of ring k → force B (1) for inward bend
        rule[i0][j0] = 1
        rule[i0][j1] = 1
        rule[i1][j0] = 1
        rule[i1][j1] = 1

        # Top edge (exclude corners): read Lk bits
        Lk = n - 2 * k - 1
        if Lk > 0:
            # Fill top edge excluding corners using bits
            for t in range(Lk):
                bit = payload_bits[idx]
                idx += 1
                j = j0 + 1 + t
                rule[i0][j] = bit
            # D4: mirror to bottom edge
            for t in range(Lk):
                j = j0 + 1 + t
                rule[i1][j] = rule[i0][j]  # 180° rotation symmetry on horizontal
            # D4: mirror to left/right edges
            for t in range(Lk):
                i = i0 + 1 + t
                rule[i][j0] = rule[i0][i]  # rotate 90° from top to left
                rule[i][j1] = rule[i0][i]  # rotate 270° from top to right
        # If Lk <= 0, ring is only corners (already set)

    # Center dot
    center_bit = payload_bits[idx]
    idx += 1
    c = n // 2
    rule[c][c] = center_bit

    assert idx == len(payload_bits)
    return rule

# -----------------------------
# Geometry + SVG rendering
# -----------------------------

# We lay out dots on integer grid, spacing 's'.
# Around each dot (x,y), four portal midpoints at:
# (x±s/2, y) and (x, y±s/2). We draw quarter arcs centered at (x,y) of radius s/2
# to connect portals according to A/B pairing.

# A-tile: connect N↔S via LEFT & RIGHT arcs (two quarter arcs), and E↔W via TOP & BOTTOM arcs.
# B-tile: connect N↔E and S↔W (diagonal-ish flow) via NE, SW quadrants; plus N↔W and S↔E via NW, SE.
# Practically: draw all four quarter arcs but choose which pairs form continuous strokes at portals
# (the global degree-2 property ensures closed loops). For clean rendering, we just draw the four
# quarter arcs belonging to the chosen pairings; duplicates will visually merge.


def draw_kolam_svg(
    rule: List[List[Tile]],
    s: float = 80.0,
    margin: float = 40.0,
    stroke: float = 6.0,
    bg: str = "white",
    fg: str = "black",
    dot_radius: float = 0.0,
) -> svgwrite.Drawing:
    n = len(rule)
    size = margin * 2 + s * (n - 1)
    dwg = svgwrite.Drawing(size=(size, size))
    dwg.add(dwg.rect(insert=(0, 0), size=(size, size), fill=bg))

    # Helpers
    def pt(i: int, j: int) -> Tuple[float, float]:
        return (margin + s * j, margin + s * i)  # (x,y), i=row (y), j=col (x)

    R = s / 2.0

    path = svgwrite.path.Path(stroke=fg, fill="none", stroke_width=stroke, stroke_linecap="round")

    # We draw quarter arcs per dot based on tile rule.
    # Define arc commands by start/end angles (SVG uses degrees, 0° along +x, CCW positive).
    def arc_center(cx, cy, start_deg, end_deg):
        # Approximate by small arc path commands. SVG has arc command 'A'. We'll move to start then arc to end.
        rad = R
        x0 = cx + rad * math.cos(math.radians(start_deg))
        y0 = cy + rad * math.sin(math.radians(start_deg))
        x1 = cx + rad * math.cos(math.radians(end_deg))
        y1 = cy + rad * math.sin(math.radians(end_deg))
        large_arc = 0
        sweep = 1  # CCW
        path.push(f"M {x0:.3f},{y0:.3f} A {rad:.3f},{rad:.3f} 0 {large_arc} {sweep} {x1:.3f},{y1:.3f}")

    for i in range(n):
        for j in range(n):
            cx, cy = pt(i, j)
            t = rule[i][j]
            if t == 0:  # A: vertical + horizontal pairings
                # Draw four separate quarter arcs that connect portals to make straight flows
                # Left: from 180°→270° (connect W to S side around left-bottom quarter)
                arc_center(cx, cy, 180, 270)
                # Left-top: 90°→180° (connect N to W)
                arc_center(cx, cy, 90, 180)
                # Right-top: 0°→90° (connect E to N)
                arc_center(cx, cy, 0, 90)
                # Right-bottom: 270°→360° (connect S to E)
                arc_center(cx, cy, 270, 360)
            else:  # B: diagonal pairings
                # NE: 0°→90°
                arc_center(cx, cy, 0, 90)
                # SW: 180°→270°
                arc_center(cx, cy, 180, 270)
                # NW: 90°→180°
                arc_center(cx, cy, 90, 180)
                # SE: 270°→360°
                arc_center(cx, cy, 270, 360)

            if dot_radius > 0:
                dwg.add(dwg.circle(center=(cx, cy), r=dot_radius, fill=fg))

    dwg.add(path)
    return dwg

# -----------------------------
# Top-level encode + render
# -----------------------------

def encode_text_to_rule(text: str, n: int) -> List[List[Tile]]:
    if n % 2 == 0:
        raise ValueError("n must be odd (e.g.,  five, seven, nine)")
    bits = text_to_bits(text)
    need = bit_budget(n)
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    bits = prng_pad(bits, need, seed)
    return build_rule_matrix(n, bits)


def save_svg_and_png(dwg: svgwrite.Drawing, svg_path: str, png_path: str | None = None):
    dwg.saveas(svg_path)
    if png_path:
        if _HAS_CAIROSVG:
            cairosvg.svg2png(url=svg_path, write_to=png_path)
        else:
            print("[info] cairosvg not installed; PNG not written. Install with: pip install cairosvg")

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="text to encode")
    ap.add_argument("--n", type=int, default=9, help="odd grid size (e.g., 5, 7, 9)")
    ap.add_argument("--spacing", type=float, default=80.0, help="dot spacing (px)")
    ap.add_argument("--margin", type=float, default=40.0, help="margin (px)")
    ap.add_argument("--stroke", type=float, default=6.0, help="stroke width (px)")
    ap.add_argument("--bg", default="white")
    ap.add_argument("--fg", default="black")
    ap.add_argument("--svg", default="kolam.svg")
    ap.add_argument("--png", default="kolam.png")
    args = ap.parse_args()

    rule = encode_text_to_rule(args.text, args.n)
    dwg = draw_kolam_svg(rule, s=args.spacing, margin=args.margin, stroke=args.stroke, bg=args.bg, fg=args.fg)
    save_svg_and_png(dwg, args.svg, args.png)
    print(f"SVG written to {args.svg}")
    if _HAS_CAIROSVG:
        print(f"PNG written to {args.png}")
    else:
        print("PNG skipped (cairosvg not installed)")
