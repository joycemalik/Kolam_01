import sys
import os

def test_svglib():
    """Test svglib which works without Cairo"""
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        
        # Create a simple SVG
        svg_content = '''<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
            <rect x="50" y="50" width="100" height="100" fill="blue" stroke="red" stroke-width="3"/>
            <text x="100" y="100" text-anchor="middle" fill="white">Hello!</text>
        </svg>'''
        
        # Write SVG file
        with open('test.svg', 'w') as f:
            f.write(svg_content)
        
        # Convert to PNG
        drawing = svg2rlg('test.svg')
        if drawing:
            renderPM.drawToFile(drawing, 'output.png', fmt='PNG')
            print("✓ svglib conversion successful!")
            return True
        else:
            print("✗ Failed to convert SVG")
            return False
            
    except Exception as e:
        print(f"✗ svglib error: {e}")
        return False

def test_pillow_simple():
    """Test Pillow for basic image creation"""
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple image
        img = Image.new('RGB', (200, 200), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill='red', outline='blue')
        draw.text((100, 100), "Pillow", fill='white', anchor='mm')
        img.save('pillow_output.png')
        print("✓ Pillow image creation successful!")
        return True
    except Exception as e:
        print(f"✗ Pillow error: {e}")
        return False

if __name__ == "__main__":
    print("Testing SVG/Image libraries...")
    print("=" * 50)
    
    # First install the working libraries
    print("Installing svglib and Pillow...")
    os.system("uv add svglib reportlab pillow")
    
    print("Testing libraries...")
    test1 = test_svglib()
    test2 = test_pillow_simple()
    
    print("=" * 50)
    if test1 or test2:
        print("✓ At least one method worked!")
    else:
        print("✗ All methods failed")