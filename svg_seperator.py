import os
import svgwrite
from lxml import etree
from copy import deepcopy

REG_MARK_SIZE = 20  # px

def extract_colors(svg_file):
    tree = etree.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Find all unique fill colors
    colors = set()
    for elem in root.xpath('//*[contains(@fill, "#")]', namespaces=ns):
        color = elem.get('fill').lower()
        if color != 'none':
            colors.add(color)
    return list(colors)

def add_registration_marks(dwg, width=1000, height=1000):
    # You can refine the width/height based on viewBox if needed
    marks = [
        ((width/2)-REG_MARK_SIZE/2, 10),  # top center
        ((width/2)-REG_MARK_SIZE/2, height - 10 - REG_MARK_SIZE),  # bottom center
    ]
    for (x, y) in marks:
        dwg.add(dwg.rect(insert=(x, y), size=(REG_MARK_SIZE, REG_MARK_SIZE), fill='black'))

def separate_color(svg_path, output_dir):
    colors = extract_colors(svg_path)
    tree = etree.parse(svg_path)
    root = tree.getroot()

    for color in colors:
        tree_copy = deepcopy(tree)
        root_copy = tree_copy.getroot()

        # Hide everything not this color
        for elem in root_copy.iter():
            fill = elem.get('fill')
            if fill and fill.lower() != color:
                elem.attrib['style'] = "display:none"
            elif fill and fill.lower() == color:
                elem.set('fill', 'black')  # set to black

        # Save new SVG
        output_path = os.path.join(output_dir, f"{color[1:]}.svg")
        with open(output_path, 'wb') as f:
            tree_copy.write(f)

        # Add registration marks to the new SVG
        dwg = svgwrite.Drawing(output_path, size=('1000px', '1000px'))
        add_registration_marks(dwg)
        dwg.saveas(output_path)
