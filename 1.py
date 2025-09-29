import cv2
import numpy as np

def generate_kolam(rows, cols, dot_spacing=50, line_thickness=2):
    """
    Generates a simple Kolam pattern.

    Args:
        rows (int): Number of rows in the dot grid.
        cols (int): Number of columns in the dot grid.
        dot_spacing (int): The space between dots.
        line_thickness (int): The thickness of the lines and dots.
    """
    # Calculate image dimensions with padding
    width = (cols + 1) * dot_spacing
    height = (rows + 1) * dot_spacing
    
    # Create a white canvas (image)
    # Using uint8 for 8-bit unsigned integer, standard for images
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # --- 1. Draw the Dot Grid (Pulli) ---
    dot_radius = line_thickness + 1
    dot_color = (0, 0, 0) # Black
    
    dot_positions = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 1) * dot_spacing
            y = (r + 1) * dot_spacing
            dot_positions.append((x, y))
            cv2.circle(image, (x, y), dot_radius, dot_color, -1) # -1 fills the circle
            
    # --- 2. Draw the Weaving Lines (Kambi) ---
    line_color = (200, 0, 0) # Red color for lines
    
    # This is a simple rule: draw four arcs around each dot.
    # This creates a basic, self-contained woven pattern.
    arc_radius = dot_spacing // 4 # The radius of the curves around the dots
    
    for (x, y) in dot_positions:
        # Arc on the top
        cv2.ellipse(image, (x, y - arc_radius), (arc_radius, arc_radius), 
                    angle=0, startAngle=0, endAngle=180, 
                    color=line_color, thickness=line_thickness)
        
        # Arc on the bottom
        cv2.ellipse(image, (x, y + arc_radius), (arc_radius, arc_radius), 
                    angle=0, startAngle=180, endAngle=360, 
                    color=line_color, thickness=line_thickness)

        # Arc on the left
        cv2.ellipse(image, (x - arc_radius, y), (arc_radius, arc_radius), 
                    angle=0, startAngle=270, endAngle=450, 
                    color=line_color, thickness=line_thickness)
                    
        # Arc on the right
        cv2.ellipse(image, (x + arc_radius, y), (arc_radius, arc_radius), 
                    angle=0, startAngle=90, endAngle=270, 
                    color=line_color, thickness=line_thickness)

    # Save the generated image
    cv2.imwrite("generated_kolam.png", image)
    print("Generated Kolam saved as 'generated_kolam.png'")

# --- Run the generator ---
# You can change the grid size here
generate_kolam(rows=5, cols=7)