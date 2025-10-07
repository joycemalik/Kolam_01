import hashlib
import random
from PIL import Image, ImageDraw, ImageFont
from typing import List

def text_to_bits(text: str) -> str:
    """1. Text -> Bits: Converts each character into an 8-bit binary string."""
    return ''.join(format(ord(char), '08b') for char in text)

def calculate_bits_needed(n: int) -> int:
    """Calculates how many unique bits are needed for a symmetric n x n grid."""
    # For an n x n grid with the specified symmetry, we only need to define
    # the pattern for the non-corner dots on one edge for each ring.
    # The number of rings is k = (n-1)/2.
    # The total number of bits is the sum of an arithmetic series:
    # (n-2) + (n-4) + ... + 1, which equals k^2.
    k = (n - 1) // 2
    return k * k

def prepare_bits(bit_string: str, length: int, seed_text: str) -> str:
    """2. Pad or Trim Bits: Adjusts the bit string to the required length."""
    if len(bit_string) >= length:
        # If too long, trim it
        return bit_string[:length]
    else:
        # If too short, pad with pseudo-random bits seeded by the text
        # This makes the padding consistent for the same input text.
        seed = int(hashlib.sha256(seed_text.encode('utf-8')).hexdigest(), 16)
        r = random.Random(seed)
        padding_needed = length - len(bit_string)
        padding = ''.join(r.choice(['0', '1']) for _ in range(padding_needed))
        return bit_string + padding

def create_grid(bits: str, n: int) -> List[List[int]]:
    """3. Bits -> Kolam Pattern: Fills the grid with A/B tiles symmetrically."""
    # 0 represents an 'A' tile (straight), 1 represents a 'B' tile (diagonal)
    grid = [[0] * n for _ in range(n)]
    bit_idx = 0
    num_rings = (n + 1) // 2

    # Fill the grid from the outermost ring to the center
    for r in range(num_rings):
        # Set corners to 'B' (1) to make the lines bend inward
        grid[r][r] = 1
        grid[r][n - 1 - r] = 1
        grid[n - 1 - r][r] = 1
        grid[n - 1 - r][n - 1 - r] = 1

        # Fill edges for all rings except the center point
        if r < num_rings - 1:
            edge_len = n - (2 * r) - 2
            edge_bits = bits[bit_idx : bit_idx + edge_len]
            bit_idx += edge_len

            if len(edge_bits) == edge_len:
                for i in range(edge_len):
                    bit = int(edge_bits[i])
                    # 4. Apply Symmetry: Copy the top edge pattern to the other three edges
                    grid[r][r + 1 + i] = bit  # Top edge
                    grid[n - 1 - r][r + 1 + i] = bit  # Bottom edge
                    grid[r + 1 + i][r] = bit  # Left edge
                    grid[r + 1 + i][n - 1 - r] = bit  # Right edge
    return grid

def draw_kolam(grid: List[List[int]], scale: int = 40, line_width: int = 4,
               bg_color: str = '#111111', line_color: str = '#FFFFFF') -> Image.Image:
    """5. Draw Kolam Lines: Renders the grid pattern into smooth, closed loops."""
    n = len(grid)
    img_size = n * scale
    image = Image.new('RGB', (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(image)
    
    half_scale = scale / 2

    for i in range(n):
        for j in range(n):
            tile_type = grid[i][j]
            
            # Define cell coordinates
            x0, y0 = j * scale, i * scale
            xc, yc = x0 + half_scale, y0 + half_scale
            x1, y1 = x0 + scale, y0 + scale

            if tile_type == 0:  # 'A' Tile: Straight connections
                draw.line([(x0, yc), (x1, yc)], fill=line_color, width=line_width) # Horizontal
                draw.line([(xc, y0), (xc, y1)], fill=line_color, width=line_width) # Vertical
            else:  # 'B' Tile: Diagonal connections
                # Arc connecting North and West points of the cell
                bbox_nw = (x0 - half_scale, y0 - half_scale, x0 + half_scale, y0 + half_scale)
                draw.arc(bbox_nw, start=0, end=90, fill=line_color, width=line_width)

                # Arc connecting South and East points of the cell
                bbox_se = (x1 - half_scale, y1 - half_scale, x1 + half_scale, y1 + half_scale)
                draw.arc(bbox_se, start=180, end=270, fill=line_color, width=line_width)

    # Use antialiasing for smoother lines
    image = image.resize((img_size // 2, img_size // 2), Image.Resampling.LANCZOS)
    return image

def generate_kolam_from_text(text: str, n: int = 25, filename: str = "kolam2.png"):
    """
    Main function to orchestrate the entire Kolam generation process.
    
    Args:
        text (str): The input text to be encoded into the Kolam.
        n (int): The size of the grid (must be an odd number).
        filename (str): The name of the output image file.
    """
    if n % 2 == 0:
        n += 1
        print(f"Grid size 'n' must be odd. Adjusting to {n}.")
    
    # --- Execute the simple algorithm flow ---
    # 1. Text -> Bits
    initial_bits = text_to_bits(text)
    
    # 2. Pad or Trim Bits
    needed_bits_count = calculate_bits_needed(n)
    final_bits = prepare_bits(initial_bits, needed_bits_count, text)
    
    # 3. Bits -> Kolam Pattern & 4. Apply Symmetry
    kolam_grid = create_grid(final_bits, n)
    
    # 5. Draw Kolam Lines
    kolam_image = draw_kolam(kolam_grid)
    
    # 6. Export Image
    kolam_image.save(filename)
    print(f"Success! Your Kolam has been saved as '{filename}'")
    
# --- Example Usage ---
if __name__ == "__main__":
    input_text = '''


import hashlib
import random
from PIL import Image, ImageDraw, ImageFont
from typing import List

def text_to_bits(text: str) -> str:
    """1. Text -> Bits: Converts each character into an 8-bit binary string."""
    return ''.join(format(ord(char), '08b') for char in text)

def calculate_bits_needed(n: int) -> int:
    """Calculates how many unique bits are needed for a symmetric n x n grid."""
    # For an n x n grid with the specified symmetry, we only need to define
    # the pattern for the non-corner dots on one edge for each ring.
    # The number of rings is k = (n-1)/2.
    # The total number of bits is the sum of an arithmetic series:
    # (n-2) + (n-4) + ... + 1, which equals k^2.
    k = (n - 1) // 2
    return k * k

def prepare_bits(bit_string: str, length: int, seed_text: str) -> str:
    """2. Pad or Trim Bits: Adjusts the bit string to the required length."""
    if len(bit_string) >= length:
        # If too long, trim it
        return bit_string[:length]
    else:
        # If too short, pad with pseudo-random bits seeded by the text
        # This makes the padding consistent for the same input text.
        seed = int(hashlib.sha256(seed_text.encode('utf-8')).hexdigest(), 16)
        r = random.Random(seed)
        padding_needed = length - len(bit_string)
        padding = ''.join(r.choice(['0', '1']) for _ in range(padding_needed))
        return bit_string + padding

def create_grid(bits: str, n: int) -> List[List[int]]:
    """3. Bits -> Kolam Pattern: Fills the grid with A/B tiles symmetrically."""
    # 0 represents an 'A' tile (straight), 1 represents a 'B' tile (diagonal)
    grid = [[0] * n for _ in range(n)]
    bit_idx = 0
    num_rings = (n + 1) // 2

    # Fill the grid from the outermost ring to the center
    for r in range(num_rings):
        # Set corners to 'B' (1) to make the lines bend inward
        grid[r][r] = 1
        grid[r][n - 1 - r] = 1
        grid[n - 1 - r][r] = 1
        grid[n - 1 - r][n - 1 - r] = 1

        # Fill edges for all rings except the center point
        if r < num_rings - 1:
            edge_len = n - (2 * r) - 2
            edge_bits = bits[bit_idx : bit_idx + edge_len]
            bit_idx += edge_len

            if len(edge_bits) == edge_len:
                for i in range(edge_len):
                    bit = int(edge_bits[i])
                    # 4. Apply Symmetry: Copy the top edge pattern to the other three edges
                    grid[r][r + 1 + i] = bit  # Top edge
                    grid[n - 1 - r][r + 1 + i] = bit  # Bottom edge
                    grid[r + 1 + i][r] = bit  # Left edge
                    grid[r + 1 + i][n - 1 - r] = bit  # Right edge
    return grid

def draw_kolam(grid: List[List[int]], scale: int = 40, line_width: int = 4,
               bg_color: str = '#111111', line_color: str = '#FFFFFF') -> Image.Image:
    """5. Draw Kolam Lines: Renders the grid pattern into smooth, closed loops."""
    n = len(grid)
    img_size = n * scale
    image = Image.new('RGB', (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(image)
    
    half_scale = scale / 2

    for i in range(n):
        for j in range(n):
            tile_type = grid[i][j]
            
            # Define cell coordinates
            x0, y0 = j * scale, i * scale
            xc, yc = x0 + half_scale, y0 + half_scale
            x1, y1 = x0 + scale, y0 + scale

            if tile_type == 0:  # 'A' Tile: Straight connections
                draw.line([(x0, yc), (x1, yc)], fill=line_color, width=line_width) # Horizontal
                draw.line([(xc, y0), (xc, y1)], fill=line_color, width=line_width) # Vertical
            else:  # 'B' Tile: Diagonal connections
                # Arc connecting North and West points of the cell
                bbox_nw = (x0 - half_scale, y0 - half_scale, x0 + half_scale, y0 + half_scale)
                draw.arc(bbox_nw, start=0, end=90, fill=line_color, width=line_width)

                # Arc connecting South and East points of the cell
                bbox_se = (x1 - half_scale, y1 - half_scale, x1 + half_scale, y1 + half_scale)
                draw.arc(bbox_se, start=180, end=270, fill=line_color, width=line_width)

    # Use antialiasing for smoother lines
    image = image.resize((img_size // 2, img_size // 2), Image.Resampling.LANCZOS)
    return image

def generate_kolam_from_text(text: str, n: int = 25, filename: str = "kolam1.png"):
    """
    Main function to orchestrate the entire Kolam generation process.
    
    Args:
        text (str): The input text to be encoded into the Kolam.
        n (int): The size of the grid (must be an odd number).
        filename (str): The name of the output image file.
    """
    if n % 2 == 0:
        n += 1
        print(f"Grid size 'n' must be odd. Adjusting to {n}.")
    
    # --- Execute the simple algorithm flow ---
    # 1. Text -> Bits
    initial_bits = text_to_bits(text)
    
    # 2. Pad or Trim Bits
    needed_bits_count = calculate_bits_needed(n)
    final_bits = prepare_bits(initial_bits, needed_bits_count, text)
    
    # 3. Bits -> Kolam Pattern & 4. Apply Symmetry
    kolam_grid = create_grid(final_bits, n)
    
    # 5. Draw Kolam Lines
    kolam_image = draw_kolam(kolam_grid)
    
    # 6. Export Image
    kolam_image.save(filename)
    print(f"Success! Your Kolam has been saved as '{filename}'")


'''
    grid_size = 21 # Try different odd numbers like 15, 25, 31...
    
    generate_kolam_from_text(input_text, n=grid_size)