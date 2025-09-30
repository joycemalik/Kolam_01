#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kolam QR — turn text into a scannable QR, rendered as kolam-style SVG.

First principles:
1) Encode text -> QR matrix (error correction keeps it robust).
2) For each dark module (tile), draw a curvy kolam motif inside the tile.
3) Add gradients, rounded caps/joins, and optional fractal recursion.
4) Keep a quiet zone and (in SAFE mode) a subtle square backdrop for scan-ability.
"""

import argparse
import math
from typing import Tuple, Dict

import segno
import svgwrite


# ---------------------------
# Palettes (simple + tasteful)
# ---------------------------
THEMES: Dict[str, Dict[str, str]] = {
    "classic": {
        "bg": "#0b0f14",      # deep midnight
        "stroke": "#f5f1e6",  # rice-flour off-white
        "accent": "#d2cbb8",
        "finder": "#f5f1e6",
        "backdrop": "#f5f1e6",
        "backdrop_opacity": "0.10",
    },
    "lotus": {
        "bg": "#0e0b10",
        "stroke": "#ffe7f0",   # soft petal
        "accent": "#f6bfd9",
        "finder": "#ffe7f0",
        "backdrop": "#ffe7f0",
        "backdrop_opacity": "0.12",
    },
    "peacock": {
        "bg": "#0a0e12",
        "stroke": "#e7fbff",   # misty cyan
        "accent": "#a7e1ff",
        "finder": "#e7fbff",
        "backdrop": "#e7fbff",
        "backdrop_opacity": "0.10",
    },
    "sunset": {
        "bg": "#0f0a0a",
        "stroke": "#fff0e0",   # warm cream
        "accent": "#ffd0a3",
        "finder": "#fff0e0",
        "backdrop": "#fff0e0",
        "backdrop_opacity": "0.12",
    },
    "mono": {
        "bg": "#0b0b0b",
        "stroke": "#ffffff",
        "accent": "#e6e6e6",
        "finder": "#ffffff",
        "backdrop": "#ffffff",
        "backdrop_opacity": "0.10",
    },
}


# --------------------------------------
# Geometry helpers (kolam motif building)
# --------------------------------------
def add_linear_gradient(dwg: svgwrite.Drawing, gid: str, c1: str, c2: str) -> None:
    lg = dwg.linearGradient(start=(0, 0), end=(1, 1), id=gid)
    lg.add_stop_color(0, c1)
    lg.add_stop_color(1, c2)
    dwg.defs.add(lg)

def motif_path(x: float, y: float, s: float, swirl: float = 0.35, rotation_deg: float = 0.0) -> str:
    """
    A tile-level kolam motif: four symmetric bezier loops that curve out
    to the midpoints and corners, then return. s is the tile size in px.
    swirl sets the curve strength (0.2-0.6 looks good).
    rotation rotates the motif to add variety.
    """
    cx, cy = x + s / 2, y + s / 2
    r = s * 0.48
    k = swirl * r

    # base points (unrotated)
    pts = [
        # Right loop
        (cx, cy - r*0.15, cx + k, cy - r*0.15, cx + r, cy, cx + k, cy + r*0.15, cx, cy + r*0.15),
        # Bottom loop
        (cx + r*0.15, cy, cx + r*0.15, cy + k, cx, cy + r, cx - r*0.15, cy + k, cx - r*0.15, cy),
        # Left loop
        (cx, cy + r*0.15, cx - k, cy + r*0.15, cx - r, cy, cx - k, cy - r*0.15, cx, cy - r*0.15),
        # Top loop
        (cx - r*0.15, cy, cx - r*0.15, cy - k, cx, cy - r, cx + r*0.15, cy - k, cx + r*0.15, cy),
    ]

    if abs(rotation_deg) > 1e-6:
        rad = math.radians(rotation_deg)
        sinr, cosr = math.sin(rad), math.cos(rad)
        def rot(px, py):
            dx, dy = px - cx, py - cy
            return (cx + dx * cosr - dy * sinr, cy + dx * sinr + dy * cosr)

        rpts = []
        for seg in pts:
            rseg = []
            for i in range(0, len(seg), 2):
                rseg.extend(rot(seg[i], seg[i+1]))
            rpts.append(tuple(rseg))
        pts = rpts

    # Build cubic beziers (M -> C -> S style path)
    d = []
    for (x0,y0, x1,y1, x2,y2, x3,y3, x4,y4) in pts:
        d.append(f"M {x0:.3f},{y0:.3f} C {x1:.3f},{y1:.3f} {x2:.3f},{y2:.3f} {x3:.3f},{y3:.3f} S {x4:.3f},{y4:.3f} {x0:.3f},{y0:.3f}")
    return " ".join(d)

def draw_fractal(dwg, group, x, y, s, depth, stroke_url, stroke_width, swirl, rot_base):
    """
    Recursively draw smaller motifs inside the tile for added richness.
    Keep it subtle to not ruin scan-ability.
    """
    if depth <= 0 or s < 6:
        return
    # four sub-quadrants
    half = s * 0.48
    offsets = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
    for i, (ox, oy) in enumerate(offsets):
        sub_x = x + (0.5 + ox) * s - half/2
        sub_y = y + (0.5 + oy) * s - half/2
        path_d = motif_path(sub_x, sub_y, half, swirl=swirl*0.95, rotation_deg=rot_base + i*15)
        group.add(
            dwg.path(
                d=path_d,
                fill="none",
                stroke=stroke_url,
                stroke_width=stroke_width * 0.72,
                stroke_linecap="round",
                stroke_linejoin="round",
                opacity=0.9
            )
        )
        draw_fractal(dwg, group, sub_x, sub_y, half, depth-1, stroke_url, stroke_width*0.72, swirl*0.95, rot_base + i*15)


# ---------------------------
# Main renderer
# ---------------------------
def render_kolam_svg(
    text: str,
    out_svg: str,
    ecc: str = "M",
    theme: str = "classic",
    module_px: float = 16.0,
    stroke_px: float = 2.4,
    fractal_depth: int = 1,
    mode: str = "safe",  # safe|art
    quiet_zone_modules: int = 4,
    swirl: float = 0.35,
):
    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}'. Choose from: {', '.join(THEMES.keys())}")

    # 1) QR matrix (auto version size scales with text length)
    qr = segno.make(text, error=ecc)
    n = qr.symbol_size()[0]  # modules including finder patterns etc.
    # segno's symbol_size returns total modules; matrix include quiet? We'll handle quiet ourselves.
    # We will iterate over qr.matrix which is n-by-n booleans.
    matrix = list(qr.matrix)  # iterable of rows of booleans
    size_modules = n + 2 * quiet_zone_modules

    # 2) Canvas size in pixels (dynamic)
    W = H = size_modules * module_px

    pal = THEMES[theme]
    bg = pal["bg"]
    stroke_col = pal["stroke"]
    accent_col = pal["accent"]
    finder_col = pal["finder"]
    backdrop_col = pal["backdrop"]
    backdrop_op = pal["backdrop_opacity"]

    dwg = svgwrite.Drawing(out_svg, size=(W, H))
    dwg.viewbox(0, 0, W, H)
    dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill=bg))

    # 3) Gradients
    add_linear_gradient(dwg, "strokeGrad", stroke_col, accent_col)
    stroke_url = "url(#strokeGrad)"

    # 4) Groups
    g_backdrop = dwg.g(id="backdrop")
    g_kolam = dwg.g(id="kolam")
    g_finders = dwg.g(id="finders")

    # 5) SAFE mode backdrop: faint rounded squares for every dark module
    if mode == "safe":
        for y_idx, row in enumerate(matrix):
            for x_idx, cell in enumerate(row):
                if cell:
                    x = (x_idx + quiet_zone_modules) * module_px
                    y = (y_idx + quiet_zone_modules) * module_px
                    g_backdrop.add(
                        dwg.rect(
                            insert=(x+0.6, y+0.6),
                            size=(module_px-1.2, module_px-1.2),
                            rx=module_px*0.18,
                            ry=module_px*0.18,
                            fill=backdrop_col,
                            opacity=backdrop_op,
                        )
                    )
        dwg.add(g_backdrop)

    # 6) Kolam strokes for every dark module
    # Add slight rotation variation to avoid a "gridy" feel: rotate by angle based on index hash.
    def rot_for(ix, iy):
        return ((ix * 37 + iy * 61) % 360) * 0.15  # small rotation

    for y_idx, row in enumerate(matrix):
        for x_idx, cell in enumerate(row):
            if not cell:
                continue
            x = (x_idx + quiet_zone_modules) * module_px
            y = (y_idx + quiet_zone_modules) * module_px

            # Main motif
            path_d = motif_path(x, y, module_px, swirl=swirl, rotation_deg=rot_for(x_idx, y_idx))
            g_kolam.add(
                dwg.path(
                    d=path_d,
                    fill="none",
                    stroke=stroke_url,
                    stroke_width=stroke_px,
                    stroke_linecap="round",
                    stroke_linejoin="round",
                )
            )

            # Fractal detail (sub-motifs)
            if fractal_depth > 0:
                draw_fractal(
                    dwg, g_kolam,
                    x, y, module_px,
                    depth=fractal_depth,
                    stroke_url=stroke_url,
                    stroke_width=stroke_px * 0.75,
                    swirl=swirl,
                    rot_base=rot_for(x_idx, y_idx)
                )

    # 7) Finder pattern embellishment (top-left, top-right, bottom-left)
    # Their coordinates in module space are (0,0), (n-7,0), (0,n-7) with size 7x7
    def draw_lotus(cx, cy, r_outer, r_inner, petals=8):
        # subtle lotus rosette over finders
        for i in range(petals):
            ang = (2*math.pi / petals) * i
            x1 = cx + r_inner * math.cos(ang)
            y1 = cy + r_inner * math.sin(ang)
            x2 = cx + r_outer * math.cos(ang + math.pi/petals)
            y2 = cy + r_outer * math.sin(ang + math.pi/petals)
            g_finders.add(
                dwg.path(
                    d=f"M {cx:.2f},{cy:.2f} Q {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f}",
                    fill="none",
                    stroke=finder_col,
                    stroke_width=stroke_px*0.9,
                    stroke_linecap="round",
                    stroke_linejoin="round",
                    opacity=0.95
                )
            )

    finder_module_size = 7
    finder_positions = [(0,0), (n - finder_module_size, 0), (0, n - finder_module_size)]
    for fx, fy in finder_positions:
        # center in px
        cx = (fx + quiet_zone_modules + finder_module_size/2) * module_px
        cy = (fy + quiet_zone_modules + finder_module_size/2) * module_px
        rO = module_px * 3.2
        rI = module_px * 1.2

        # underlay: faint rounded square to highlight the finder region
        g_finders.add(
            dwg.rect(
                insert=((fx + quiet_zone_modules) * module_px, (fy + quiet_zone_modules) * module_px),
                size=(finder_module_size * module_px, finder_module_size * module_px),
                rx=module_px*0.6, ry=module_px*0.6,
                fill="none",
                stroke=finder_col,
                stroke_width=stroke_px*0.8,
                opacity=0.25
            )
        )
        draw_lotus(cx, cy, rO, rI, petals=10)

    dwg.add(g_finders)
    dwg.add(g_kolam)
    dwg.save()


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Kolam QR SVG generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="Encode text into a kolam-style QR SVG")
    e.add_argument("--text", required=True, help="Input text to encode")
    e.add_argument("--out", required=True, help="Output SVG path")
    e.add_argument("--ecc", default="M", choices=list("LMQH"), help="QR error correction level")
    e.add_argument("--theme", default="classic", choices=list(THEMES.keys()))
    e.add_argument("--module", type=float, default=16.0, help="Pixels per module (tile size)")
    e.add_argument("--stroke", type=float, default=2.4, help="Stroke width (px)")
    e.add_argument("--fractal", type=int, default=1, help="Fractal depth (0=off)")
    e.add_argument("--mode", default="safe", choices=["safe", "art"], help="safe keeps a scannable backdrop")
    e.add_argument("--quiet", type=int, default=4, help="Quiet zone size in modules")
    e.add_argument("--swirl", type=float, default=0.35, help="Curve strength of motif (0.2-0.6 good)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.cmd == "encode":
        render_kolam_svg(
            text=args.text,
            out_svg=args.out,
            ecc=args.ecc,
            theme=args.theme,
            module_px=args.module,
            stroke_px=args.stroke,
            fractal_depth=args.fractal,
            mode=args.mode,
            quiet_zone_modules=args.quiet,
            swirl=args.swirl,
        )
        print(f"✅ Wrote {args.out}")

if __name__ == "__main__":
    main()
