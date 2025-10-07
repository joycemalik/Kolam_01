import hashlib
import random
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw
import svgwrite

# --- Step 1: Text to Bits ---
def text_to_bits(text: str) -> str:
    """Converts the entire text into a single UTF-8 binary stream."""
    encoded_text = text.encode('utf-8')
    return ''.join(format(byte, '08b') for byte in encoded_text)

# --- Step 2: Pick Grid Size Automatically ---
def get_grid_capacity(n: int) -> int:
    """Calculates the bit capacity of an n x n grid with the specified structure."""
    if n < 5 or n % 2 == 0:
        return 0
    # This formula accounts for the bits needed for the diamond shape and center.
    # It's derived from summing the unique elements per ring in a D4-symmetric diamond.
    k = (n - 1) // 2
    return 2 * k * (k - 1) + 1

def determine_grid_size(bit_count: int) -> int:
    """Finds the smallest odd grid size 'n' that can hold the bitstream."""
    n = 5
    while get_grid_capacity(n) < bit_count:
        n += 2
    return n

# --- Step 3: Fit Bits to Capacity ---
def prepare_bits(text: str) -> Tuple[str, int]:
    """Pads the bitstream to fit the automatically determined grid size."""
    bit_string = text_to_bits(text)
    n = determine_grid_size(len(bit_string))
    capacity = get_grid_capacity(n)
    
    if len(bit_string) < capacity:
        # Seed a PRNG with the text's hash for deterministic padding
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        r = random.Random(seed)
        padding_needed = capacity - len(bit_string)
        padding = ''.join(r.choice(['0', '1']) for _ in range(padding_needed))
        bit_string += padding
        
    return bit_string, n

# --- Step 4: Bits -> Tiles (with Beauty Pass) ---
def create_kolam_grid(bits: str, n: int) -> List[List[Optional[int]]]:
    """Creates the final grid with all automatic shaping rules applied."""
    grid: List[List[Optional[int]]] = [[None] * n for _ in range(n)]
    bit_idx = 0
    center = (n - 1) // 2

    # Fill ring by ring, from outside in
    num_rings = (n + 1) // 2
    for r in range(num_rings - 1): # All rings except the center
        ring_edge_len = center - r
        
        # Take bits for the top-right edge of the ring
        bits_for_edge = bits[bit_idx : bit_idx + (ring_edge_len -1)]
        bit_idx += len(bits_for_edge)
        
        for i in range(ring_edge_len):
            tile = int(bits_for_edge[i-1]) if i > 0 else 1 # Corners are always B
            
            # Apply D4 Symmetry to place the tile in 8 spots
            grid[r + i][center + ring_edge_len - i] = tile # Top-right
            grid[r + i][center - ring_edge_len + i] = tile # Top-left
            grid[n - 1 - r - i][center + ring_edge_len - i] = tile # Bottom-right
            grid[n - 1 - r - i][center - ring_edge_len + i] = tile # Bottom-left
            
            grid[center + ring_edge_len - i][r + i] = tile # Right-top
            grid[center - ring_edge_len + i][r + i] = tile # Left-top
            grid[center + ring_edge_len - i][n - 1 - r - i] = tile # Right-bottom
            grid[center - ring_edge_len + i][n - 1 - r - i] = tile # Left-bottom

    # Apply the "Beauty Pass" modifications
    
    # 1. Center Rosette: Overwrite the 3x3 center
    if n >= 5:
        grid[center][center] = 0  # Center is 'A'
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                grid[center + i][center + j] = 1 # Ring is 'B'
    else: # For a 3x3 grid
         grid[center][center] = int(bits[-1])

    # 2. Teardrop Tips (Cap Tiles): Overwrite the 4 diamond tips
    # A cap tile connects the incoming line back on itself in a curl.
    # We'll use tile type '2' for North-South caps and '3' for East-West caps.
    grid[0][center] = 2 # North
    grid[n-1][center] = 2 # South
    grid[center][0] = 3 # West
    grid[center][n-1] = 3 # East
    
    return grid

# --- Step 5 & 6: Render to PNG and SVG ---
def render_kolam(grid: List[List[Optional[int]]], text: str, scale: int = 40):
    """Renders the final grid to both PNG and SVG files."""
    n = len(grid)
    bg_color = '#0c0c0c'
    line_color = '#fafafa'
    line_width = max(1, int(scale / 12))

    # --- PNG Rendering (with Pillow) ---
    png_size = n * scale
    image = Image.new('RGB', (png_size, png_size), bg_color)
    draw = ImageDraw.Draw(image)
    
    # --- SVG Rendering (with svgwrite) ---
    svg_filename = f'kolam_{text.replace(" ", "_")}.svg'
    dwg = svgwrite.Drawing(svg_filename, size=(f'{png_size}px', f'{png_size}px'))
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=bg_color))
    
    for r in range(n):
        for c in range(n):
            tile = grid[r][c]
            if tile is None:
                continue

            # Define cell coordinates
            x0, y0 = c * scale, r * scale
            xc, yc = x0 + scale / 2, y0 + scale / 2
            x1, y1 = x0 + scale, y0 + scale
            hs = scale / 2

            # Tile drawing logic
            if tile == 0:  # 'A' Tile (straight)
                draw.line([(x0, yc), (x1, yc)], fill=line_color, width=line_width)
                draw.line([(xc, y0), (xc, y1)], fill=line_color, width=line_width)
                dwg.add(dwg.path(d=f"M {x0},{yc} H {x1} M {xc},{y0} V {y1}", stroke=line_color, stroke_width=line_width, fill='none'))
            elif tile == 1:  # 'B' Tile (diagonal)
                draw.arc((x0 - hs, y0 - hs, x0 + hs, y0 + hs), 0, 90, fill=line_color, width=line_width)
                draw.arc((x1 - hs, y1 - hs, x1 + hs, y1 + hs), 180, 270, fill=line_color, width=line_width)
                dwg.add(dwg.path(d=f"M {xc},{y0} A {hs},{hs} 0 0 0 {x0},{yc}", stroke=line_color, stroke_width=line_width, fill='none'))
                dwg.add(dwg.path(d=f"M {x1},{yc} A {hs},{hs} 0 0 0 {xc},{y1}", stroke=line_color, stroke_width=line_width, fill='none'))
            elif tile == 2: # 'Cap' Tile (North-South curl)
                draw.arc((xc-hs, y0, xc+hs, y0+scale), 180, 0, fill=line_color, width=line_width)
                dwg.add(dwg.path(d=f"M {xc-hs},{yc} A {hs/2},{hs} 0 0 1 {xc+hs},{yc}", stroke=line_color, stroke_width=line_width, fill='none'))
            elif tile == 3: # 'Cap' Tile (East-West curl)
                draw.arc((x0, yc-hs, x0+scale, yc+hs), 270, 90, fill=line_color, width=line_width)
                dwg.add(dwg.path(d=f"M {xc},{yc-hs} A {hs},{hs/2} 0 0 1 {xc},{yc+hs}", stroke=line_color, stroke_width=line_width, fill='none'))

    # Save files
    png_filename = f'kolam_{text.replace(" ", "_")}.png'
    image = image.resize((png_size // 2, png_size // 2), Image.Resampling.LANCZOS)
    image.save(png_filename)
    dwg.save()
    print(f"✅ Success! Kolam saved as '{png_filename}' and '{svg_filename}'")

# --- Main Execution ---
if __name__ == "__main__":
    input_text ='''Thats it.Insize, symmetry, tips, rosette, paddi automatically from the text and its hash.Ou'''
    
    # 1. & 2. & 3. Get bits and grid size automatically
    payload, grid_size = prepare_bits(input_text)
    
    print(f"Input text: '{input_text}'")
    print(f"Bit count: {len(text_to_bits(input_text))}, Padded to: {len(payload)}")
    print(f"Required grid size: {grid_size}x{grid_size}")
    
    # 4. Create the final grid structure
    final_grid = create_kolam_grid(payload, grid_size)
    
    # 5. & 6. Render the final image
    render_kolam(final_grid, input_text)