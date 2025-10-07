import argparse, base64, binascii, json, math, os, sys
from typing import Tuple, List
import numpy as np
import cv2
import svgwrite
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import hashlib
from math import cos, sin, pi
import io

# =========================
# Robust SVG Support Detection
# =========================
CAIROSVG_AVAILABLE = False
SVGLIB_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
    print("✅ CairoSVG available")
except (ImportError, OSError) as e:
    CAIROSVG_AVAILABLE = False
    if "cairo" in str(e).lower():
        print("⚠️  CairoSVG not available (Cairo library missing)")
    else:
        print(f"⚠️  CairoSVG not available: {e}")

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    SVGLIB_AVAILABLE = True
    print("✅ SVGLib available")
except ImportError as e:
    SVGLIB_AVAILABLE = False
    print(f"⚠️  SVGLib not available: {e}")

def get_supported_formats() -> List[str]:
    """Return list of supported image formats"""
    basic_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif']
    return basic_formats

def print_format_info():
    """Print information about supported formats"""
    formats = get_supported_formats()
    print("Supported image formats:")
    for fmt in formats:
        print(f"  ✅ {fmt}")
    print("\n📝 Note: This version outputs PNG format for maximum compatibility")

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
# AUTHENTIC SIKKU KOLAM ENCODING
# =========================

class KolamPoint:
    """Represents a point in the kolam dot grid (pulli)"""
    def __init__(self, x, y, row, col):
        self.x = x
        self.y = y
        self.row = row
        self.col = col
        self.connections = []  # Connected points
        self.visited = False

class SikkuKolamEncoder:
    """Encode data using authentic sikku kolam principles"""
    
    def __init__(self, grid_size, dot_spacing):
        self.grid_size = grid_size
        self.dot_spacing = dot_spacing
        self.pulli_grid = self.create_pulli_grid()
        self.path = []  # Single continuous path
        
    def create_pulli_grid(self):
        """Create traditional dot grid (pulli) foundation"""
        grid = {}
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.dot_spacing
                y = row * self.dot_spacing
                point = KolamPoint(x, y, row, col)
                grid[(row, col)] = point
                
        # Connect adjacent dots
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                point = grid[(row, col)]
                # 8-directional connections for authentic kolam
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            point.connections.append(grid[(nr, nc)])
        
        return grid
    
    def encode_text_to_path_decisions(self, text):
        """Convert text to path decisions for sikku kolam"""
        binary = self.text_to_binary(text)
        decisions = []
        
        # Group binary into movement instructions
        for i in range(0, len(binary), 3):
            chunk = binary[i:i+3] if i+2 < len(binary) else binary[i:] + [0] * (3 - len(binary[i:]))
            
            # Map 3-bit patterns to kolam path decisions
            pattern = (chunk[0] << 2) | (chunk[1] << 1) | chunk[2]
            
            if pattern == 0:    # 000
                decisions.append('straight')
            elif pattern == 1:  # 001  
                decisions.append('turn_left_60')
            elif pattern == 2:  # 010
                decisions.append('turn_right_60')
            elif pattern == 3:  # 011
                decisions.append('small_loop')
            elif pattern == 4:  # 100
                decisions.append('medium_loop')
            elif pattern == 5:  # 101
                decisions.append('large_loop')
            elif pattern == 6:  # 110
                decisions.append('sikku_cross')  # Traditional crossing
            else:               # 111
                decisions.append('flower_motif') # Decorative element
                
        return decisions
    
    def text_to_binary(self, text):
        """Convert text to binary list"""
        binary = []
        for char in text.encode('utf-8'):
            for i in range(7, -1, -1):
                binary.append((char >> i) & 1)
        return binary
    
    def create_sikku_kolam(self, text):
        """Create authentic sikku kolam from text"""
        decisions = self.encode_text_to_path_decisions(text)
        
        # Start from center of grid
        center_row = self.grid_size // 2
        center_col = self.grid_size // 2
        current_point = self.pulli_grid[(center_row, center_col)]
        
        # Initial direction (0=right, 1=down-right, 2=down, etc.)
        direction = 0
        path_segments = []
        
        for decision in decisions:
            segment = self.execute_path_decision(current_point, direction, decision)
            if segment:
                path_segments.extend(segment['points'])
                current_point = segment['end_point']
                direction = segment['new_direction']
        
        return path_segments
    
    def execute_path_decision(self, current_point, direction, decision):
        """Execute a single path decision in the kolam"""
        points = []
        
        if decision == 'straight':
            # Continue in current direction
            next_point = self.get_point_in_direction(current_point, direction)
            if next_point:
                points = self.create_smooth_curve(current_point, next_point)
                return {
                    'points': points,
                    'end_point': next_point,
                    'new_direction': direction
                }
        
        elif decision == 'turn_left_60':
            # Turn 60 degrees left (sikku kolam standard turn)
            new_direction = (direction - 1) % 6  # 60-degree increments
            next_point = self.get_point_in_direction(current_point, new_direction)
            if next_point:
                points = self.create_smooth_curve(current_point, next_point)
                return {
                    'points': points,
                    'end_point': next_point,
                    'new_direction': new_direction
                }
        
        elif decision == 'turn_right_60':
            # Turn 60 degrees right
            new_direction = (direction + 1) % 6
            next_point = self.get_point_in_direction(current_point, new_direction)
            if next_point:
                points = self.create_smooth_curve(current_point, next_point)
                return {
                    'points': points,
                    'end_point': next_point,
                    'new_direction': new_direction
                }
        
        elif decision in ['small_loop', 'medium_loop', 'large_loop']:
            # Create traditional kolam loops
            loop_size = {'small_loop': 2, 'medium_loop': 3, 'large_loop': 4}[decision]
            loop_points = self.create_traditional_loop(current_point, direction, loop_size)
            return {
                'points': loop_points,
                'end_point': current_point,  # Returns to start after loop
                'new_direction': direction
            }
        
        elif decision == 'sikku_cross':
            # Traditional sikku crossing pattern
            cross_points = self.create_sikku_crossing(current_point, direction)
            return {
                'points': cross_points,
                'end_point': current_point,
                'new_direction': (direction + 2) % 6  # 120-degree turn
            }
        
        elif decision == 'flower_motif':
            # Traditional flower decoration
            flower_points = self.create_flower_motif(current_point)
            return {
                'points': flower_points,
                'end_point': current_point,
                'new_direction': direction
            }
        
        return None
    
    def get_point_in_direction(self, point, direction):
        """Get the next point in a given direction (0-5 for 60-degree increments)"""
        # Direction mappings for hexagonal grid movement
        direction_deltas = [
            (0, 1),   # Right
            (1, 1),   # Down-right
            (1, 0),   # Down
            (0, -1),  # Left
            (-1, -1), # Up-left
            (-1, 0)   # Up
        ]
        
        dr, dc = direction_deltas[direction]
        new_row = point.row + dr
        new_col = point.col + dc
        
        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            return self.pulli_grid[(new_row, new_col)]
        return None
    
    def create_smooth_curve(self, start_point, end_point):
        """Create smooth bezier curve between two pulli points"""
        import math
        
        # Calculate control points for authentic kolam curve
        dx = end_point.x - start_point.x
        dy = end_point.y - start_point.y
        
        # Control points create the characteristic kolam flow
        control1_x = start_point.x + dx * 0.3
        control1_y = start_point.y + dy * 0.3
        control2_x = end_point.x - dx * 0.3
        control2_y = end_point.y - dy * 0.3
        
        # Generate smooth curve points
        points = []
        num_segments = 10
        
        for i in range(num_segments + 1):
            t = i / num_segments
            
            # Cubic bezier calculation
            x = ((1-t)**3 * start_point.x + 
                 3*(1-t)**2*t * control1_x + 
                 3*(1-t)*t**2 * control2_x + 
                 t**3 * end_point.x)
            y = ((1-t)**3 * start_point.y + 
                 3*(1-t)**2*t * control1_y + 
                 3*(1-t)*t**2 * control2_y + 
                 t**3 * end_point.y)
            
            points.append((x, y))
        
        return points
    
    def create_traditional_loop(self, center_point, direction, size):
        """Create traditional kolam loop around pulli dots"""
        import math
        
        loop_points = []
        radius = size * self.dot_spacing * 0.4
        
        # Create circular loop with traditional proportions
        for i in range(24):  # 24 points for smooth circle
            angle = i * 2 * math.pi / 24
            x = center_point.x + radius * math.cos(angle)
            y = center_point.y + radius * math.sin(angle)
            loop_points.append((x, y))
        
        # Close the loop
        loop_points.append(loop_points[0])
        
        return loop_points
    
    def create_sikku_crossing(self, center_point, direction):
        """Create traditional sikku (knot) crossing pattern"""
        # Traditional figure-8 crossing
        import math
        
        cross_points = []
        scale = self.dot_spacing * 0.3
        
        # Create figure-8 pattern (traditional sikku)
        for i in range(32):
            t = i * 2 * math.pi / 32
            
            # Parametric figure-8
            x = center_point.x + scale * math.sin(t)
            y = center_point.y + scale * math.sin(t) * math.cos(t)
            
            cross_points.append((x, y))
        
        return cross_points
    
    def create_flower_motif(self, center_point):
        """Create traditional flower motif"""
        import math
        
        flower_points = []
        
        # 6-petaled flower (traditional kolam motif)
        for petal in range(6):
            petal_angle = petal * math.pi / 3
            
            # Each petal as a small curve
            for i in range(5):
                t = i / 4
                radius = self.dot_spacing * 0.2 * (1 - t)
                angle = petal_angle + t * math.pi / 6
                
                x = center_point.x + radius * math.cos(angle)
                y = center_point.y + radius * math.sin(angle)
                flower_points.append((x, y))
        
        return flower_points
def create_kolam_png(grid_bits: List[int], N: int, output_path: str = "kolam.png", 
                    tile_size: int = 40, margin: int = 50, line_width: int = 3, mode: str = "traditional"):
    """
    Create kolam PNG - traditional for compatibility or sikku for authenticity
    """
    # Calculate image size
    image_size = margin * 2 + N * tile_size
    
    # Create image with white background
    img = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw border
    draw.rectangle([2, 2, image_size-2, image_size-2], outline='black', width=2)
    
    if mode == "sikku":
        # Use authentic sikku kolam approach
        kolam_size = image_size - 2 * margin
        dot_spacing = kolam_size // (N + 2)  # +2 for border spacing
        
        encoder = SikkuKolamEncoder(N + 4, dot_spacing)  # Larger grid for authentic patterns
        
        # Convert bits back to text for encoding
        text = decode_bits_to_text(grid_bits)
        
        # Create authentic sikku kolam path
        kolam_path = encoder.create_sikku_kolam(text)
        
        # Draw traditional pulli (dot grid) foundation
        draw_pulli_grid(draw, encoder.pulli_grid, margin, dot_radius=2)
        
        # Draw the continuous sikku kolam line
        draw_continuous_kolam_path(draw, kolam_path, margin, line_width)
        
        # Add traditional corner decorations
        draw_corner_decorations(draw, image_size, margin, line_width)
    else:
        # Use traditional tile-based approach (compatible with decoder)
        tile_data = bits_to_tile_grid(grid_bits, N, tile_size, margin)
        
        # Draw traditional kolam tiles with sikku patterns
        draw_continuous_sikku_kolam(draw, tile_data, tile_size, line_width)
    
    # Save the image
    img.save(output_path, 'PNG', dpi=(300, 300))
    return output_path

def bits_to_tile_grid(grid_bits: List[int], N: int, tile_size: int, margin: int):
    """Convert bits back to tile grid format for traditional rendering"""
    tile_data = []
    bit_idx = 0
    
    # Skip header bits for simplicity 
    bit_idx = 8 * 7  # Skip header
    
    for row in range(N):
        tile_row = []
        for col in range(N):
            if bit_idx + 2 < len(grid_bits):
                rot_bit1 = grid_bits[bit_idx] if bit_idx < len(grid_bits) else 0
                rot_bit2 = grid_bits[bit_idx + 1] if bit_idx + 1 < len(grid_bits) else 0
                dot_bit = grid_bits[bit_idx + 2] if bit_idx + 2 < len(grid_bits) else 0
                bit_idx += 3
                
                rotation = (rot_bit1 << 1) | rot_bit2
                x = margin + col * tile_size
                y = margin + row * tile_size
                
                tile_row.append({
                    'x': x,
                    'y': y,
                    'rot': rotation,
                    'dot': dot_bit
                })
            else:
                # Default tile if we run out of bits
                x = margin + col * tile_size
                y = margin + row * tile_size
                tile_row.append({
                    'x': x,
                    'y': y,
                    'rot': 0,
                    'dot': 0
                })
        tile_data.append(tile_row)
    
    return tile_data

def decode_bits_to_text(grid_bits):
    """Convert bits back to text for authentic encoding"""
    # Skip header and extract payload
    try:
        header_bits = grid_bits[:8*7]
        header = bits_to_bytes(header_bits)
        version, n_tiles, text_len = unpack_header(header)
        
        payload_start = 8 * 7
        payload_bits = grid_bits[payload_start:payload_start + text_len * 8]
        payload_bytes = bits_to_bytes(payload_bits)
        
        return payload_bytes.decode('utf-8', errors='replace')
    except:
        # Fallback: use first few bits as simple text
        return "Sikku Kolam"

def draw_pulli_grid(draw, pulli_grid, margin, dot_radius=2):
    """Draw traditional pulli (dot) grid foundation"""
    for point in pulli_grid.values():
        x = margin + point.x
        y = margin + point.y
        draw.ellipse([x - dot_radius, y - dot_radius, 
                     x + dot_radius, y + dot_radius], fill='black')

def draw_continuous_kolam_path(draw, path_points, margin, line_width):
    """Draw the single continuous sikku kolam line"""
    if len(path_points) < 2:
        return
    
    # Offset points by margin
    offset_points = [(margin + x, margin + y) for x, y in path_points]
    
    # Draw smooth continuous line
    for i in range(len(offset_points) - 1):
        draw.line([offset_points[i], offset_points[i + 1]], 
                 fill='black', width=line_width)
    
    # Add rounded end caps for traditional appearance
    if len(offset_points) >= 2:
        cap_radius = line_width // 2
        for point in [offset_points[0], offset_points[-1]]:
            draw.ellipse([point[0] - cap_radius, point[1] - cap_radius,
                         point[0] + cap_radius, point[1] + cap_radius], 
                        fill='black')

def draw_corner_decorations(draw, image_size, margin, line_width):
    """Add traditional kolam corner decorations"""
    import math
    
    corner_size = margin // 3
    
    # Four corners with traditional motifs
    corners = [
        (margin // 2, margin // 2),                    # Top-left
        (image_size - margin // 2, margin // 2),       # Top-right
        (image_size - margin // 2, image_size - margin // 2),  # Bottom-right
        (margin // 2, image_size - margin // 2)        # Bottom-left
    ]
    
    for cx, cy in corners:
        # Traditional triangular corner motif
        for i in range(6):
            angle = i * math.pi / 3
            x1 = cx + corner_size * math.cos(angle)
            y1 = cy + corner_size * math.sin(angle)
            x2 = cx + corner_size * 0.5 * math.cos(angle + math.pi / 6)
            y2 = cy + corner_size * 0.5 * math.sin(angle + math.pi / 6)
            
            draw.line([(cx, cy), (x1, y1)], fill='black', width=1)
            draw.line([(x1, y1), (x2, y2)], fill='black', width=1)



def draw_continuous_sikku_kolam(draw, tile_data, tile_size, line_width):
    """
    Draw simplified continuous sikku kolam patterns
    """
    N = len(tile_data)
    
    for row in range(N):
        for col in range(N):
            tile = tile_data[row][col]
            x, y = tile['x'], tile['y']
            center_x = x + tile_size // 2
            center_y = y + tile_size // 2
            margin = tile_size // 4
            
            # Draw curved patterns based on rotation
            if tile['rot'] == 0:
                # Vertical flowing curves
                draw.arc([x + margin, y, x + tile_size - margin, y + tile_size],
                         90, 270, fill='black', width=line_width)
            elif tile['rot'] == 1:
                # Horizontal flowing curves  
                draw.arc([x, y + margin, x + tile_size, y + tile_size - margin],
                         0, 180, fill='black', width=line_width)
            elif tile['rot'] == 2:
                # Diagonal curves
                draw.arc([x, y, x + tile_size, y + tile_size],
                         45, 225, fill='black', width=line_width)
            else:  # rot == 3
                # Reverse diagonal curves
                draw.arc([x, y, x + tile_size, y + tile_size],
                         135, 315, fill='black', width=line_width)
            
            # Draw dots
            if tile['dot'] == 1:
                dot_radius = max(3, tile_size // 12)
                draw.ellipse([center_x - dot_radius, center_y - dot_radius,
                             center_x + dot_radius, center_y + dot_radius], 
                            fill='black')

def draw_kolam_tile(draw, x: int, y: int, size: int, rotation: int, dot_bit: int, line_width: int):
    """
    Draw a single kolam tile using PIL
    """
    center_x = x + size // 2
    center_y = y + size // 2
    radius = int(size * 0.4)
    
    # Define the four connection points
    points = [
        (x, center_y),           # Left
        (center_x, y),           # Top
        (x + size, center_y),    # Right
        (center_x, y + size)     # Bottom
    ]
    
    # Apply rotation
    rotated_points = []
    for px, py in points:
        # Translate to origin
        dx = px - center_x
        dy = py - center_y
        
        # Rotate (90 degrees * rotation)
        for _ in range(rotation):
            dx, dy = -dy, dx
            
        # Translate back
        rotated_points.append((center_x + dx, center_y + dy))
    
    left, top, right, bottom = rotated_points
    
    # Draw arcs based on rotation
    center = (center_x, center_y)
    
    if rotation == 0:
        # Left to Top, Right to Bottom
        draw_arc(draw, left, top, center, radius, line_width)
        draw_arc(draw, right, bottom, center, radius, line_width)
    elif rotation == 1:
        # Top to Right, Bottom to Left
        draw_arc(draw, top, right, center, radius, line_width)
        draw_arc(draw, bottom, left, center, radius, line_width)
    elif rotation == 2:
        # Right to Bottom, Left to Top
        draw_arc(draw, right, bottom, center, radius, line_width)
        draw_arc(draw, left, top, center, radius, line_width)
    else:  # rotation == 3
        # Bottom to Left, Top to Right
        draw_arc(draw, bottom, left, center, radius, line_width)
        draw_arc(draw, top, right, center, radius, line_width)
    
    # Draw dot if needed
    if dot_bit == 1:
        dot_radius = max(2, size // 15)
        draw.ellipse([center_x - dot_radius, center_y - dot_radius,
                     center_x + dot_radius, center_y + dot_radius], 
                    fill='black')

def draw_arc(draw, start: Tuple[int, int], end: Tuple[int, int], 
            center: Tuple[int, int], radius: int, line_width: int):
    """
    Draw a curved arc between two points like traditional kolam patterns
    Creates quarter-circle arcs that are characteristic of kolam geometry
    """
    sx, sy = start
    ex, ey = end
    cx, cy = center
    
    import math
    
    # Calculate the angle between start and end points relative to center
    start_angle = math.atan2(sy - cy, sx - cx)
    end_angle = math.atan2(ey - cy, ex - cx)
    
    # Convert to degrees for PIL's arc function
    start_deg = math.degrees(start_angle)
    end_deg = math.degrees(end_angle)
    
    # Ensure we draw the shorter arc (quarter circle for kolam style)
    angle_diff = end_deg - start_deg
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360
    
    # Adjust for proper kolam quarter-circle arcs
    if abs(angle_diff) > 90:
        # If more than 90 degrees, draw the complementary shorter arc
        if angle_diff > 0:
            end_deg = start_deg + 90
        else:
            end_deg = start_deg - 90
    
    # Create bounding box for the arc centered at the tile center
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    
    # Draw the quarter-circle arc using PIL's arc function
    try:
        # For traditional kolam look, ensure smooth quarter circles
        if start_deg > end_deg:
            start_deg, end_deg = end_deg, start_deg
            
        draw.arc(bbox, start_deg, end_deg, fill='black', width=line_width)
        
        # Add rounded end caps for better appearance
        cap_radius = line_width // 2
        if cap_radius > 0:
            # Start cap
            start_x = cx + radius * math.cos(math.radians(start_deg))
            start_y = cy + radius * math.sin(math.radians(start_deg))
            draw.ellipse([start_x - cap_radius, start_y - cap_radius,
                         start_x + cap_radius, start_y + cap_radius], fill='black')
            
            # End cap
            end_x = cx + radius * math.cos(math.radians(end_deg))
            end_y = cy + radius * math.sin(math.radians(end_deg))
            draw.ellipse([end_x - cap_radius, end_y - cap_radius,
                         end_x + cap_radius, end_y + cap_radius], fill='black')
            
    except Exception:
        # Fallback: draw with line segments if arc fails
        num_segments = 12
        points = []
        
        for i in range(num_segments + 1):
            t = i / num_segments
            angle_rad = math.radians(start_deg + t * (end_deg - start_deg))
            arc_x = cx + radius * math.cos(angle_rad)
            arc_y = cy + radius * math.sin(angle_rad)
            points.append((arc_x, arc_y))
        
        # Draw smooth curve with line segments
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill='black', width=line_width)

# =========================
# Robust Image Loading
# =========================
def load_image_robust(img_path: str) -> np.ndarray:
    """
    Robust image loader for PNG, JPG, etc.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    file_ext = os.path.splitext(img_path)[1].lower()
    
    if file_ext not in get_supported_formats():
        raise ValueError(f"Unsupported format: {file_ext}. Supported: {get_supported_formats()}")
    
    try:
        # Try OpenCV first
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            print(f"✅ Loaded {file_ext} via OpenCV - Shape: {img.shape}")
            return _convert_to_grayscale(img)
        
        # Fallback to PIL
        try:
            pil_img = Image.open(img_path)
            if pil_img.mode == 'RGBA':
                # Convert RGBA to RGB
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[-1])
                pil_img = background
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
                
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            print(f"✅ Loaded {file_ext} via PIL - Shape: {img.shape}")
            return _convert_to_grayscale(img)
        except Exception as pil_err:
            print(f"❌ PIL failed: {pil_err}")
            
    except Exception as e:
        raise RuntimeError(f"Error loading image {img_path}: {e}")
    
    raise RuntimeError(f"Failed to load image: {img_path}")

def _convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    print(f"Grayscale dimensions: {gray.shape}")
    return gray

# =========================
# Decoder Functions
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
    Decode kolam pattern from PNG image
    """
    gray = load_image_robust(img_path)
    
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
    Ns = [N_hint] if N_hint else [5, 7, 10, 15, 21, 25, 29, 33]
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
# Encoding: text -> bits
# =========================
def build_bitstream_from_text(text: str, N: int) -> List[int]:
    payload = text.encode("utf-8")
    h = pack_header(N*N, len(payload), version=1)
    chk = crc32(payload).to_bytes(4, 'big')
    full = h + payload + chk
    return bytes_to_bits(full)

# =========================
# Main encode/decode
# =========================
def encode_cmd(args):
    text = args.text
    N = args.grid if args.grid else auto_grid_for_text(len(text.encode("utf-8")), bits_per_tile=3)

    # Build bitstream
    bits = build_bitstream_from_text(text, N)

    # Generate PNG
    output_path = args.out
    if not output_path.lower().endswith('.png'):
        output_path += '.png'

    png_path = create_kolam_png(
        bits, N, 
        output_path=output_path,
        tile_size=args.tile_size,
        margin=args.margin,
        line_width=args.line_width
    )

    print(f"✅ Saved: {png_path}")
    print(f"📊 Grid: {N}x{N}, capacity bits: {N*N*3}, used bits: {len(bits)}")
    print(f"🖼️  Image size: {args.margin * 2 + N * args.tile_size}px")

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
        print("-" * 40)
        print(text)
        print("-" * 40)
        print(f"\n📊 Meta: {meta}")
        print(f"🎯 Image format: {file_ext}")
    except Exception as e:
        print(f"❌ Decode failed: {e}")
        if "border" in str(e).lower():
            print(f"\n💡 Try using --debug to see detection images")
            print(f"💡 Ensure the image has a clear black border around the pattern")

def formats_cmd(args):
    print_format_info()

def build_parser():
    p = argparse.ArgumentParser(description="KolamCode PNG: encode/decode with PNG format")
    sub = p.add_subparsers()

    e = sub.add_parser("encode", help="encode text -> Kolam PNG")
    e.add_argument("--text", required=True, help="text to encode")
    e.add_argument("--out", default="kolam.png", help="output PNG path")
    e.add_argument("--grid", type=int, default=None, help="grid size N (omit for auto)")
    e.add_argument("--tile-size", type=int, default=40, help="tile size in pixels")
    e.add_argument("--margin", type=int, default=50, help="quiet margin in pixels")
    e.add_argument("--line-width", type=int, default=3, help="line width for drawing")
    e.set_defaults(func=encode_cmd)

    d = sub.add_parser("decode", help="decode PNG -> text")
    d.add_argument("--image", required=True, help="input PNG image")
    d.add_argument("--grid", type=int, default=None, help="optional hint for N")
    d.add_argument("--debug", action="store_true", help="save debug images showing detection process")
    d.set_defaults(func=decode_cmd)
    
    f = sub.add_parser("formats", help="list supported image formats")
    f.set_defaults(func=formats_cmd)
    
    return p

def auto_grid_for_text(text_len: int, bits_per_tile: int = 3, minN: int = 5, maxN: int = 61) -> int:
    """Choose N so N*N*bits_per_tile >= header(7*8) + text*8 + crc(32)."""
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
    print("🔍 KolamCode PNG - No Cairo Dependencies Needed")
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
        import traceback
        if getattr(args, 'debug', False):
            traceback.print_exc()
        sys.exit(1)