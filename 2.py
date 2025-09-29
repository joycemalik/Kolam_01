import cv2
import numpy as np

def detect_dots(image_path):
    """
    Detects the dots (pulli) in an image of a Kolam.

    Args:
        image_path (str): The path to the Kolam image file.
    """
    # --- 1. Load and Pre-process the Image ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a blur to reduce noise and help detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # --- 2. Detect Circles using Hough Circle Transform ---
    # This function is good for finding circles.
    # param1 (canny threshold) and param2 (accumulator threshold) are key.
    # You MUST tune these values for your specific image for good results.
    detected_circles = cv2.HoughCircles(blurred,
                                       cv2.HOUGH_GRADIENT,
                                       dp=1.2,          # Inverse ratio of resolution
                                       minDist=30,      # Minimum distance between detected centers
                                       param1=100,      # Upper threshold for the internal Canny edge detector
                                       param2=25,       # Threshold for center detection.
                                       minRadius=3,     # Minimum circle radius
                                       maxRadius=15)    # Maximum circle radius

    # --- 3. Draw the Detected Circles on the Original Image ---
    if detected_circles is not None:
        # Convert the circle parameters (x, y, radius) to integers
        detected_circles = np.uint16(np.around(detected_circles))
        
        print(f"Detected {len(detected_circles[0, :])} dots.")

        for pt in detected_circles[0, :]:
            x, y, r = pt[0], pt[1], pt[2]
            
            # Draw the circumference of the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 2) # Green circle
            
            # Draw a small circle (dot) at the center
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3) # Red center
    else:
        print("No dots were detected. Try adjusting the detection parameters.")

    # Save the result image
    cv2.imwrite("detected_dots.png", image)
    print("Detection result saved as 'detected_dots.png'")


# --- Run the detector ---
# Make sure you have an image file named 'kolam_to_detect.png'
# or change the path below.
detect_dots("generated_kolam.png")