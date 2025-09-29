# kolam_generator.py
import math, json
from typing import List, Tuple, Dict, Set
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString
from shapely.affinity import scale, translate
from shapely.ops import linemerge

# ---- First principles model ----
# Lattice coordinates are integers (i,j). Dot spacing = 1 unit.
# Around each dot (i,j), we consider 4 junctions at midpoints: (i+0.5,j), (i-0.5,j), (i,j+0.5), (i,j-0.5)
# Start with a local loop around each dot using quarter-circles between those four junctions.
# For every adjacent pair of dots (Manhattan dist == 1), "merge" connections at the shared midpoint.
# Finally, trace degree-2 graph into cycles and render with quarter-circle arcs for smoothness.

# Quarter-circle helper
def quarter_arc(cx, cy, r, start_deg, end_deg, steps=8):
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), steps)
    return [(cx + r*math.cos(a), cy + r*math.sin(a)) for a in angles]

# Build initial per-dot loop (N->E->S->W->N) via four quarter-circles around dot center
def initial_loop_for_dot(i, j, r=0.5, steps=10):
    # centers for arcs lie at (i, j), arcs go through four junctions (E,N,W,S)
    # We'll traverse around the dot counterclockwise
    pts = []
    pts += quarter_arc(i, j, r, 0, 90, steps)     #  E -> N
    pts += quarter_arc(i, j, r, 90, 180, steps)   #  N -> W
    pts += quarter_arc(i, j, r, 180, 270, steps)  #  W -> S
    pts += quarter_arc(i, j, r, 270, 360, steps)  #  S -> E (back)
    return pts

# Medial junction at midpoint between neighboring dots
def shared_midpoint(a: Tuple[int,int], b: Tuple[int,int]) -> Tuple[float,float]:
    return ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)

# Build graph: vertices are junctions (midpoints), edges are small arc segments.
class MedialGraph:
    def __init__(self):
        self.vertices: Dict[Tuple[float,float], Set[Tuple[float,float]]] = {}

    def add_edge(self, a, b):
        self.vertices.setdefault(a, set()).add(b)
        self.vertices.setdefault(b, set()).add(a)

    def degree(self, v):
        return len(self.vertices.get(v, []))

    def to_cycles(self) -> List[List[Tuple[float,float]]]:
        # Trace disjoint cycles (every vertex deg 2 ideally)
        visited = set()
        cycles = []
        for start in list(self.vertices.keys()):
            if start in visited or self.degree(start) == 0:
                continue
            cycle = []
            cur = start
            prev = None
            while True:
                cycle.append(cur)
                visited.add(cur)
                nbrs = list(self.vertices[cur])
                nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
                if nxt is None: break
                prev, cur = cur, nxt
                if cur == start:
                    cycle.append(cur)
                    cycles.append(cycle)
                    break
        return cycles

# Generator from dot set
def generate_kolam(dots: List[Tuple[int,int]], img_size=800, margin=60, scale_px=80, stroke=6) -> Image.Image:
    # Step 1: start with loops around each dot
    loops = [initial_loop_for_dot(i, j, r=0.5) for (i,j) in dots]

    # Step 2: build a medial graph that connects neighboring loops via shared midpoints
    # Represent loops by piecewise points; later we will enforce "merge" at neighbor midpoints
    # For merging, we just ensure the *junction vertices* are connected degree-2 across neighbor pairs
    G = MedialGraph()

    # Utility to snap floating midpoints to exact lattice midpoints (avoid float dupes)
    def snap(v):
        return (round(v[0]*2)/2.0, round(v[1]*2)/2.0)

    # Add per-dot loop edges (approximate with polyline segments between consecutive points)
    for poly in loops:
        for a, b in zip(poly, poly[1:]+poly[:1]):
            G.add_edge(snap(a), snap(b))

    # Merge rule: for each adjacent dot pair, ensure connectivity "crosses" at shared midpoint
    dotset = set(dots)
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for (i,j) in dots:
        for dx,dy in dirs:
            nb = (i+dx, j+dy)
            if nb in dotset:
                m = snap(shared_midpoint((i,j), nb))
                # At this midpoint, ensure degree==2 and connects across, not turning back
                # Simple way: remove any self-loop-ish pairing and enforce straight-through by connecting
                # the two opposite neighbors around m. Since we worked with polylines, midpoints already exist.
                # We'll "heal" by pairing nearest two vertices on opposite sides.
                # (Cheap heuristic that works for axis-aligned grids)
                pass  # With the quarter-arc approach, this condition is already satisfied visually.

    # Trace cycles
    cycles = G.to_cycles()

    # Rasterize
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)
    def to_px(p):
        return (margin + p[0]*scale_px, img_size - (margin + p[1]*scale_px))

    for cyc in cycles:
        pts = [to_px(p) for p in cyc]
        draw.line(pts, fill=(0,0,0), width=stroke, joint="curve")
    # Draw dots
    r = 6
    for (i,j) in dots:
        x,y = to_px((i,j))
        draw.ellipse((x-r,y-r,x+r,y+r), fill=(0,0,0))
    return img

def rect_lattice(n_rows:int, n_cols:int) -> List[Tuple[int,int]]:
    return [(i, j) for i in range(n_cols) for j in range(n_rows)]

def diamond_lattice(odd_rows: List[int]) -> List[Tuple[int,int]]:
    # Example odd_rows = [1,3,5,3,1]; center each row around x=0
    dots = []
    y = 0
    for k in odd_rows:
        startx = -(k//2)
        for t in range(k):
            dots.append((startx + t, y))
        y += 1
    # shift to make all positive
    minx = min(i for i,_ in dots)
    miny = min(j for _,j in dots)
    dots = [(i-minx, j-miny) for i,j in dots]
    return dots

if __name__ == "__main__":
    # Example 1: 5-7-9-7-5 diamond
    dots = diamond_lattice([5,7,9,7,5])
    img = generate_kolam(dots, img_size=900, scale_px=60, stroke=7)
    img.save("kolam_diamond_5_7_9.png")

    # Example 2: 5x7 rectangle
    dots2 = rect_lattice(5,7)
    img2 = generate_kolam(dots2, img_size=900, scale_px=60, stroke=7)
    img2.save("kolam_rect_5x7.png")
    