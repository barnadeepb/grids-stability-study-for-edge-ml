import sys
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

for svg_path in sys.argv[1:]:
    png_path = svg_path.rsplit(".", 1)[0] + ".png"
    drawing = svg2rlg(svg_path)
    renderPM.drawToFile(drawing, png_path, fmt="PNG", dpi=300)
    print(f"{svg_path} -> {png_path}")
