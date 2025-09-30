import turtle
import math

# --- Default Configuration ---
# These can be overridden by user input.
DEFAULT_CONFIG = {
    "grid_size": 15,          # N x N grid of dots (must be odd for perfect center)
    "dot_spacing": 30,        # Distance between each dot
    "dot_size": 4,            # Radius of the dots
    "line_width": 1.5,
    "drawing_speed": 0,       # 0 is the fastest, 1-10 are progressively slower
    "dot_color": "black",
    "color_layer_1": "#003366", # Star (Dark Blue)
    "color_layer_2": "#006400", # Crisscross (Dark Green)
    "color_layer_3": "#FF8C00", # Diamond (Dark Orange)
    "color_layer_4": "#8A2BE2", # Petals (Blue Violet)
    "draw_layer_1": True,     # Draw the main star
    "draw_layer_2": True,     # Draw the crisscross pattern
    "draw_layer_3": True,     # Draw the central diamond
    "draw_layer_4": True,     # Draw the outer petal loops
}

# --- Utility Functions ---

def get_user_config():
    """Gets drawing parameters from the user, using defaults if input is empty."""
    print("--- Configure Your Kolam ---")
    print("Press Enter to use the default value shown in [brackets].\n")
    
    config = DEFAULT_CONFIG.copy()
    
    try:
        # Get numerical inputs
        grid_size_in = input(f"Grid Size (odd number) [{config['grid_size']}]: ")
        if grid_size_in:
            config['grid_size'] = int(grid_size_in)
        
        # Ensure grid size is odd for symmetry
        if config['grid_size'] % 2 == 0:
            config['grid_size'] += 1
            print(f"Grid size must be odd. Adjusted to {config['grid_size']}.")

        # Get color inputs
        config['dot_color'] = input(f"Dot Color [{config['dot_color']}]: ") or config['dot_color']
        config['color_layer_1'] = input(f"Layer 1 Color (Star) [{config['color_layer_1']}]: ") or config['color_layer_1']
        config['color_layer_2'] = input(f"Layer 2 Color (Crisscross) [{config['color_layer_2']}]: ") or config['color_layer_2']
        config['color_layer_3'] = input(f"Layer 3 Color (Diamond) [{config['color_layer_3']}]: ") or config['color_layer_3']
        config['color_layer_4'] = input(f"Layer 4 Color (Petals) [{config['color_layer_4']}]: ") or config['color_layer_4']
        
        # Get boolean inputs for drawing layers
        def get_bool_input(prompt, default):
            val = input(f"{prompt} (y/n) [{'y' if default else 'n'}]: ").lower()
            if val == 'y': return True
            if val == 'n': return False
            return default
            
        config['draw_layer_1'] = get_bool_input("Draw Layer 1 (Star)?", config['draw_layer_1'])
        config['draw_layer_2'] = get_bool_input("Draw Layer 2 (Crisscross)?", config['draw_layer_2'])
        config['draw_layer_3'] = get_bool_input("Draw Layer 3 (Diamond)?", config['draw_layer_3'])
        config['draw_layer_4'] = get_bool_input("Draw Layer 4 (Petals)?", config['draw_layer_4'])

    except ValueError:
        print("\nInvalid input. Using default values.")
        return DEFAULT_CONFIG
        
    print("\nConfiguration set. Starting generator...")
    return config

def setup_screen(config):
    """Sets up the turtle screen based on configuration."""
    screen_dim = config['grid_size'] * config['dot_spacing'] + 100
    screen = turtle.Screen()
    screen.setup(width=screen_dim, height=screen_dim)
    screen.bgcolor("white")
    screen.title("Complex Kolam Generator")
    return screen

def get_dot_coords(row, col, config, start_pos):
    """Calculates the absolute (x, y) coordinates for a dot index."""
    x = start_pos['x'] + col * config['dot_spacing']
    y = start_pos['y'] - row * config['dot_spacing']
    return x, y

def draw_dots(t, config, start_pos):
    """Draws the grid of dots (pulli)."""
    t.color(config['dot_color'])
    t.penup()
    for row in range(config['grid_size']):
        for col in range(config['grid_size']):
            t.goto(get_dot_coords(row, col, config, start_pos))
            t.dot(config['dot_size'])

# --- Pattern Drawing Functions ---

def draw_layer_1_star(t, config, start_pos, center_idx):
    """Draws a large star pattern connecting perimeter points."""
    t.color(config['color_layer_1'])
    t.penup()

    for i in range(config['grid_size']):
        # Connect top edge to bottom edge
        t.goto(get_dot_coords(0, i, config, start_pos))
        t.pendown()
        t.goto(get_dot_coords(config['grid_size'] - 1, (i + center_idx) % config['grid_size'], config, start_pos))
        t.penup()

        # Connect left edge to right edge
        t.goto(get_dot_coords(i, 0, config, start_pos))
        t.pendown()
        t.goto(get_dot_coords((i + center_idx) % config['grid_size'], config['grid_size'] - 1, config, start_pos))
        t.penup()

def draw_layer_2_crisscross(t, config, start_pos):
    """Draws an intersecting diagonal crisscross pattern."""
    t.color(config['color_layer_2'])
    t.penup()
    
    gs = config['grid_size']
    for i in range(1, gs):
        # Top-left to bottom-right diagonals
        t.goto(get_dot_coords(0, i, config, start_pos))
        t.pendown()
        t.goto(get_dot_coords(i, 0, config, start_pos))
        t.penup()
        
        # Bottom-right to top-left diagonals
        t.goto(get_dot_coords(gs - 1, i, config, start_pos))
        t.pendown()
        t.goto(get_dot_coords(i, gs - 1, config, start_pos))
        t.penup()
        
        # Top-right to bottom-left diagonals
        t.goto(get_dot_coords(0, i, config, start_pos))
        t.pendown()
        t.goto(get_dot_coords(gs - 1 - i, gs - 1, config, start_pos))
        t.penup()
        
        t.goto(get_dot_coords(i, 0, config, start_pos))
        t.pendown()
        t.goto(get_dot_coords(gs-1, gs - 1 - i, config, start_pos))
        t.penup()

def draw_layer_3_diamond(t, config, start_pos, center_idx):
    """Draws a series of concentric diamonds around the center."""
    t.color(config['color_layer_3'])
    t.width(config['line_width'] + 1) # Make this central element bolder
    t.penup()
    
    for i in range(1, center_idx + 1):
        p_top = get_dot_coords(center_idx - i, center_idx, config, start_pos)
        p_right = get_dot_coords(center_idx, center_idx + i, config, start_pos)
        p_bottom = get_dot_coords(center_idx + i, center_idx, config, start_pos)
        p_left = get_dot_coords(center_idx, center_idx - i, config, start_pos)

        t.goto(p_top)
        t.pendown()
        t.goto(p_right)
        t.goto(p_bottom)
        t.goto(p_left)
        t.goto(p_top)
        t.penup()
    t.width(config['line_width']) # Reset width

def draw_layer_4_petal_loops(t, config, start_pos):
    """Draws intricate petal-like loops around the outer dots."""
    t.color(config['color_layer_4'])
    t.penup()
    
    gs = config['grid_size']
    for i in range(0, gs - 1):
        # Top edge loops
        p1 = get_dot_coords(0, i, config, start_pos)
        p2 = get_dot_coords(1, i + 1, config, start_pos)
        t.goto(p1)
        t.pendown()
        t.setheading(t.towards(p2) + 90)
        t.circle(config['dot_spacing'] * 0.7, 180)
        t.penup()

        # Left edge loops
        p1 = get_dot_coords(i, 0, config, start_pos)
        p2 = get_dot_coords(i+1, 1, config, start_pos)
        t.goto(p1)
        t.pendown()
        t.setheading(t.towards(p2) - 90)
        t.circle(-config['dot_spacing'] * 0.7, 180)
        t.penup()
        
        # Bottom edge loops
        p1 = get_dot_coords(gs - 1, i, config, start_pos)
        p2 = get_dot_coords(gs - 2, i + 1, config, start_pos)
        t.goto(p1)
        t.pendown()
        t.setheading(t.towards(p2) - 90)
        t.circle(-config['dot_spacing'] * 0.7, 180)
        t.penup()

        # Right edge loops
        p1 = get_dot_coords(i, gs - 1, config, start_pos)
        p2 = get_dot_coords(i + 1, gs - 2, config, start_pos)
        t.goto(p1)
        t.pendown()
        t.setheading(t.towards(p2) + 90)
        t.circle(config['dot_spacing'] * 0.7, 180)
        t.penup()
        
# --- Main Execution ---

def main():
    """Main function to run the Kolam generator."""
    
    # 1. Get configuration from user
    config = get_user_config()
    
    # 2. Setup screen and turtle
    screen = setup_screen(config)
    pen = turtle.Turtle()
    pen.speed(config['drawing_speed'])
    pen.hideturtle()
    pen.width(config['line_width'])
    
    # 3. Calculate grid properties
    gs = config['grid_size']
    ds = config['dot_spacing']
    start_pos = {
        'x': -((gs - 1) * ds) / 2,
        'y': ((gs - 1) * ds) / 2
    }
    center_idx = (gs - 1) // 2

    # 4. Draw the Kolam
    print("\nStage 1: Drawing the Pulli (Dot Grid)...")
    draw_dots(pen, config, start_pos)
    
    if config['draw_layer_4']:
        print("Stage 2: Drawing Layer 4 (Petals)...")
        draw_layer_4_petal_loops(pen, config, start_pos)
        
    if config['draw_layer_1']:
        print("Stage 3: Drawing Layer 1 (Star)...")
        draw_layer_1_star(pen, config, start_pos, center_idx)
    
    if config['draw_layer_2']:
        print("Stage 4: Drawing Layer 2 (Crisscross)...")
        draw_layer_2_crisscross(pen, config, start_pos)

    if config['draw_layer_3']:
        print("Stage 5: Drawing Layer 3 (Central Diamond)...")
        draw_layer_3_diamond(pen, config, start_pos, center_idx)

    print("\n--- Kolam Generation Complete! ---")
    screen.exitonclick()

if __name__ == "__main__":
    main()

