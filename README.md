# KolamCode: Traditional Kolam QR Code Generator

KolamCode transforms text into beautiful QR codes using traditional South Indian kolam patterns. It combines the functionality of QR codes with the aesthetic beauty of kolam art, creating scannable codes that look like traditional geometric patterns.

## Features

- **Multi-format Support**: Generate and decode PNG, JPG, SVG, WEBP, BMP images
- **Three Visual Styles**: Basic, Fancy, and Clean rendering options
- **Color Palettes**: 5 beautiful color schemes inspired by traditional art
- **Customizable Dots**: Adjustable transparency for kolam dots (pulli)
- **Auto-sizing**: Automatically calculates optimal grid size for your text
- **Traditional Geometry**: Authentic kolam curves and patterns
- **High Performance**: Optimized rendering with SVG output

## Quick Start

### Installation

Ensure you have Python 3.7+ installed, then install required dependencies:

```bash
pip install numpy opencv-python svgwrite pillow
```

For SVG support (optional but recommended):
```bash
pip install cairosvg
# OR
pip install svglib reportlab
```

### Basic Usage

```bash
# Encode text to kolam QR code
python kolamcode1.py encode --text "Hello World"

# Decode kolam QR code back to text
python kolamcode1.py decode --image kolam.svg
```

## Detailed Usage

### Encoding (Text → Kolam QR)

```bash
python kolamcode1.py encode [OPTIONS]
```

#### Required Parameters
- `--text TEXT`: The text you want to encode

#### Optional Parameters
- `--out FILENAME`: Output file path (default: `kolam.svg`)
- `--style STYLE`: Visual style - `basic`, `fancy`, or `clean` (default: `fancy`)
- `--palette PALETTE`: Color scheme - `basic`, `indigo`, `vermilion`, `jade`, `royal` (default: `basic`)
- `--grid N`: Grid size NxN (auto-calculated if omitted)
- `--tile-px N`: Tile size in pixels (default: 42)
- `--margin-px N`: Border margin in pixels (default: 180)
- `--dot-opacity FLOAT`: Dot transparency 0.0-1.0 (varies by palette)
- `--fractal-depth N`: Micro-patterns inside tiles 0-2 (fancy style only)

#### Examples

```bash
# Simple kolam with default settings
python kolamcode1.py encode --text "Traditional Art"

# Fancy style with custom colors
python kolamcode1.py encode --text "Beautiful Kolam" --style fancy --palette indigo

# Clean style with custom dot opacity
python kolamcode1.py encode --text "Minimal Design" --style clean --dot-opacity 0.7

# Large format with custom sizing
python kolamcode1.py encode --text "Exhibition Piece" --tile-px 60 --margin-px 200

# Specific grid size
python kolamcode1.py encode --text "Precise Layout" --grid 25 --out my_kolam.svg
```

### Decoding (Image → Text)

```bash
python kolamcode1.py decode [OPTIONS]
```

#### Required Parameters
- `--image FILENAME`: Input image file (PNG, JPG, SVG, WEBP, BMP)

#### Optional Parameters
- `--grid N`: Grid size hint (auto-detected if omitted)
- `--debug`: Save debug images showing detection process

#### Examples

```bash
# Decode any supported image format
python kolamcode1.py decode --image kolam.svg
python kolamcode1.py decode --image screenshot.png
python kolamcode1.py decode --image photo.jpg

# Debug mode to troubleshoot detection issues
python kolamcode1.py decode --image problematic.png --debug

# With grid size hint for better accuracy
python kolamcode1.py decode --image kolam.svg --grid 21
```

### Format Information

```bash
# Check supported image formats
python kolamcode1.py formats
```

## Visual Styles

### 1. Basic Style
- **Best For**: High contrast, easy scanning, printing
- **Features**: Simple black lines on white background
- **Palette**: Grey dots with black lines
- **Use Case**: Documents, business cards, functional QR codes

```bash
python kolamcode1.py encode --text "Scan Me" --style basic
```

### 2. Fancy Style (Default)
- **Best For**: Artistic display, traditional aesthetics
- **Features**: Gradient ropes, glowing effects, traditional kolam geometry
- **Palettes**: 5 color schemes with cultural significance
- **Use Case**: Art installations, cultural displays, decorative purposes

```bash
python kolamcode1.py encode --text "Art Gallery" --style fancy --palette royal
```

### 3. Clean Style
- **Best For**: Modern minimalism, sophisticated patterns
- **Features**: Complex geometry with simple black lines and dots
- **Benefits**: Traditional complexity without visual noise
- **Use Case**: Modern design, architectural applications, contemporary art

```bash
python kolamcode1.py encode --text "Modern Tradition" --style clean
```

## Color Palettes

### Basic (Default)
- **Colors**: Black lines on white, grey dots
- **Best For**: Maximum contrast and readability
- **Dot Opacity**: 0.6 (moderate visibility)

### Indigo
- **Colors**: Deep blue tones with silver highlights
- **Inspiration**: Night sky and moonlight
- **Dot Opacity**: 0.85 (prominent dots)

### Vermilion
- **Colors**: Sacred red with warm earth tones
- **Inspiration**: Traditional tilaka and temple art
- **Dot Opacity**: 0.85 (traditional visibility)

### Jade
- **Colors**: Natural green with fresh highlights
- **Inspiration**: Sacred tulsi and prosperity
- **Dot Opacity**: 0.85 (nature-inspired)

### Royal
- **Colors**: Purple and gold with regal elegance
- **Inspiration**: Traditional royal court art
- **Dot Opacity**: 0.85 (majestic presence)

## Advanced Features

### Auto-sizing
The system automatically calculates the optimal grid size based on your text length:
- **Short text** (< 50 chars): Smaller grids (21x21)
- **Medium text** (50-200 chars): Medium grids (25x25, 29x29)
- **Long text** (> 200 chars): Larger grids (up to 61x61)

### Dot Opacity Control
Fine-tune the visibility of kolam dots (pulli):
```bash
# Subtle dots
python kolamcode1.py encode --text "Subtle" --dot-opacity 0.3

# Prominent dots
python kolamcode1.py encode --text "Bold" --dot-opacity 0.9
```

### Custom Sizing
Adjust the physical dimensions:
```bash
# Large format for printing
python kolamcode1.py encode --text "Print Me" --tile-px 80 --margin-px 300

# Compact design
python kolamcode1.py encode --text "Small" --tile-px 30 --margin-px 100
```

### Fractal Details (Fancy Style Only)
Add micro-patterns within tiles:
```bash
# Level 1: Subtle inner patterns
python kolamcode1.py encode --text "Detailed" --fractal-depth 1

# Level 2: Complex micro-kolams
python kolamcode1.py encode --text "Intricate" --fractal-depth 2
```

## File Format Support

### Encoding Output
- **SVG**: Vector format, infinite scalability (recommended)
- **PNG**: Raster format (requires additional libraries)

### Decoding Input
- **SVG**: Vector graphics (requires cairosvg or svglib)
- **PNG, JPG, JPEG**: Standard raster formats
- **WEBP**: Modern web format
- **BMP, TIFF**: Legacy formats
- **GIF**: Animated format support

## Troubleshooting

### Decoding Issues

1. **Border Detection Problems**:
   ```bash
   # Use debug mode to see detection process
   python kolamcode1.py decode --image problem.png --debug
   ```

2. **SVG Conversion Issues**:
   ```bash
   # Install SVG support
   pip install cairosvg
   ```

3. **Low Quality Images**:
   - Ensure image has clear black border
   - Use high resolution (recommended: 1000+ pixels)
   - Avoid compression artifacts

### Encoding Tips

1. **Choose Appropriate Style**:
   - Use `basic` for maximum scannability
   - Use `fancy` for artistic display
   - Use `clean` for modern aesthetics

2. **Optimize for Use Case**:
   - **Digital display**: SVG format, fancy style
   - **Printing**: Basic style, high tile-px
   - **Mobile scanning**: Clean style, medium size

3. **Text Length Guidelines**:
   - **Short URLs**: Perfect for all styles
   - **Sentences**: Use clean or fancy style
   - **Paragraphs**: Consider basic style for reliability

## Examples Gallery

### Cultural Heritage Display
```bash
python kolamcode1.py encode \
  --text "Traditional kolam patterns have been passed down through generations" \
  --style fancy --palette vermilion --tile-px 50 --margin-px 200 \
  --out heritage_display.svg
```

### Modern Business Card
```bash
python kolamcode1.py encode \
  --text "https://mywebsite.com" \
  --style clean --dot-opacity 0.4 \
  --out business_card.svg
```

### Art Installation
```bash
python kolamcode1.py encode \
  --text "Where tradition meets technology" \
  --style fancy --palette royal --fractal-depth 2 \
  --tile-px 70 --out installation.svg
```

### Educational Material
```bash
python kolamcode1.py encode \
  --text "Learn about kolam geometry and sacred patterns" \
  --style basic --grid 25 \
  --out educational.svg
```

## Technical Details

### Algorithm
- **Encoding**: Text → UTF-8 → Bits → Grid rotation/dot pattern
- **Error Correction**: CRC32 checksum for data integrity
- **Grid Mapping**: 3 bits per tile (2 for rotation, 1 for dot)
- **Border Detection**: Automatic perspective correction

### Performance
- **Generation**: Milliseconds for typical text
- **File Size**: SVG files typically 50-500KB
- **Scalability**: Vector format scales to any size
- **Memory**: Efficient processing of large grids

### Compatibility
- **Python**: 3.7+ required
- **Operating Systems**: Windows, macOS, Linux
- **Browsers**: All modern browsers support SVG
- **Printers**: Vector format ideal for high-quality printing

## Contributing

KolamCode combines computer science with traditional art. Contributions welcome for:
- Additional color palettes inspired by regional art traditions
- New geometric patterns based on different kolam styles
- Performance optimizations
- Enhanced error correction algorithms
- Cultural authenticity improvements

---

*KolamCode bridges ancient wisdom with modern technology, creating QR codes that honor the mathematical beauty and cultural significance of traditional kolam art.*