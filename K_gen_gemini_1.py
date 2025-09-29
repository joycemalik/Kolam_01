import turtle
import math

# --- Complex Parameter Configuration ---
# These parameters control the complexity and shape of the pattern
# Experiment with them to get drastically different designs!

PARAMS = {
    'n1': 3.0,  # Controls overall size and roundness
    'n2': 5.0,  # Affects the 'star' shape
    'n3': 0.6,  # Affects the sharpness and 'spikiness'
    'm': 50,    # **Crucial: Number of rotational symmetries (petals/arms)**
    'a': 1.0,   # Scaling factor
    'b': 2.0,   # Scaling factor
    'r_scale': 100 # Overall radius/size    
}

DRAWING_DETAIL = 360  # Number of points to plot (higher = smoother curve)
LINE_COLOR = "purple"
BACKGROUND_COLOR = "black"
DRAWING_SPEED = 0

# --- Parametric Function (Simplified Superformula logic) ---
def complex_radius(theta, p):
    """
    Calculates the radial distance (radius) for a given angle (theta)
    based on the complex parameters 'p'.
    """
    # Simplified Superformula-like calculation for complex radial shape
    m = p['m']
    n1, n2, n3 = p['n1'], p['n2'], p['n3']
    a, b = p['a'], p['b']
    
    # Calculate the term inside the power function
    term1 = (math.cos(m * theta / 4) / a)**n2
    term2 = (math.sin(m * theta / 4) / b)**n3
    
    # Check for division by zero risk (though math.pow handles it well usually)
    if n1 == 0:
         return 0
         
    # The radial formula itself
    r = p['r_scale'] * (abs(term1) + abs(term2))**(-1 / n1)
    return r

# --- Drawing Function ---
def draw_complex_kolam(t, p, detail):
    """Draws the Kolam based on the complex parameters."""
    t.up() # Lift the pen to move to the starting point

    # Start the loop from 0 to 360 degrees
    for i in range(detail + 1):
        # Convert step to radians
        angle_rad = math.radians(i)
        
        # Calculate the radial distance using the complex function
        r = complex_radius(angle_rad, p)
        
        # Convert polar coordinates (r, angle) to Cartesian (x, y)
        x = r * math.cos(angle_rad)
        y = r * math.sin(angle_rad)
        
        # Move the turtle
        if i == 0:
            t.goto(x, y)
            t.down() # Put pen down to start drawing
        else:
            t.goto(x, y)

# --- Setup and Execution ---
if __name__ == "__main__":
    # Setup the Turtle Screen
    screen = turtle.Screen()
    screen.setup(width=700, height=700)
    screen.bgcolor(BACKGROUND_COLOR)
    screen.title("Complex Parametric Kolam Generator")

    # Setup the Turtle Pen
    pen = turtle.Turtle()
    pen.speed(DRAWING_SPEED)
    pen.color(LINE_COLOR)
    pen.width(2)
    pen.hideturtle()

    print(f"Generating complex Kolam with m={PARAMS['m']} symmetry...")
    
    # Draw the pattern
    draw_complex_kolam(pen, PARAMS, DRAWING_DETAIL)

    print("Complex Kolam generation complete!")

    # Keep the window open
    turtle.done()