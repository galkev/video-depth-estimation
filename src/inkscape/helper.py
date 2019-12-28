import sys
sys.path.append('/snap/inkscape/4693/share/inkscape/extensions')

# We will use the inkex module with the predefined Effect base class.
import inkex
# The simplestyle module provides functions for style parsing.
from simplestyle import *

import simpletransform


import numpy as np


def compute_bounds_union(bounds_list):
    p = np.array(bounds_list).reshape(-1, 2)
    bb_min = np.array([np.min(p[:, 0]), np.min(p[:, 1])])
    bb_max = np.array([np.max(p[:, 0]), np.max(p[:, 1])])

    return np.array([bb_min, bb_max])


def point_center(p):
    return np.mean(p, axis=0)


def exp_decreasing_range(start, count, div=2):
    return [start / (div ** i) for i in range(count)]


def mirror_y(p, y):
    if isinstance(y, list):
        y = y[1]
    return np.array([p[0], p[1] + 2 * (y - p[1])])


def points_to_svg(points, flip_y=True, close=False):
    assert all(len(p) == 2 for p in points)

    points_str = ["{} {}".format(p[0], p[1] if not flip_y else -p[1]) for p in points]
    svg = "M " + " L ".join(points_str)

    if close:
        svg += " Z"

    return svg


def create_path(points, arrow=False, flip_y=True, stroke="black", stroke_width=0.2):
    style = {
        'stroke': stroke,
        'stroke-width': str(stroke_width),
        'fill': 'none'
    }

    if arrow:
        style["marker-end"] = "url(#Arrow1Lend)"

    svg = points_to_svg(points, flip_y=flip_y, close=False)

    line_attribs = {
        'style': formatStyle(style),
        inkex.addNS('label', 'inkscape'): "path",
        'd': svg
    }

    # line = inkex.etree.SubElement(parent, inkex.addNS('path', 'svg'), line_attribs)
    line = inkex.etree.Element(inkex.addNS('path', 'svg'), line_attribs)
    return line


def create_path_rectangle(p1, p2, fill, flip_y=True):
    points = [p1, np.array([p2[0], p1[1]]), p2, np.array([p1[0], p2[1]])]

    assert len(points) == 4

    return create_polygon(points, fill=fill, flip_y=flip_y)


def create_polygon(points, fill, stroke, flip_y=True):
    style = {
        'stroke': stroke,
        'stroke-width': '0.3',
        'fill': fill,
        'stroke-linejoin': 'round'
    }

    svg = points_to_svg(points, flip_y=flip_y, close=True)

    line_attribs = {
        'style': formatStyle(style),
        inkex.addNS('label', 'inkscape'): "path",
        'd': svg
    }

    # line = inkex.etree.SubElement(parent, inkex.addNS('path', 'svg'), line_attribs)
    line = inkex.etree.Element(inkex.addNS('path', 'svg'), line_attribs)
    return line


def rotate_point(point, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    return np.dot(R, point)


def get_box_points(dim, angle, pos=(0, 0), center_y=True):
    l = np.array(pos)
    u = l + np.array([dim[0], dim[1]]) + rotate_point(np.array([dim[2], 0]), angle)

    """
     65
    324
    01
    """

    x = np.array([
        l, l + np.array([dim[0], 0]), l + np.array([dim[0], dim[1]]), l + np.array([0, dim[1]]),
        u - np.array([0, dim[1]]), u, u - np.array([dim[0], 0])
    ])

    if center_y:
        x[:, 1] -= (point_center(x[[1, 5]]) - x[0])[1]

    return x


def get_frustum_points(dim, angle, pos=(0, 0), center_y=True):
    b1 = get_box_points([dim[0], dim[1], dim[2]], angle, pos, center_y)
    b2 = get_box_points([dim[0], dim[3], dim[4]], angle, pos, center_y)

    ind1 = [0, 3, 6]
    ind2 = [1, 2, 4, 5]

    frustum = np.zeros_like(b1)
    frustum[ind1] = b1[ind1]
    frustum[ind2] = b2[ind2]

    return frustum


def create_box_from_points(layer, points, colors):
    """
     65
    324
    01
    """

    sides = [
        points[[0, 1, 2, 3]]
    ]

    if points[2, 0] != points[5, 0]:
        sides.extend([
            points[[1, 4, 5, 2]],
            points[[2, 5, 6, 3]]
        ])

        layer = inkex.etree.SubElement(layer, 'g')

    for s, c in zip(sides, colors[:len(sides)]):
        layer.append(create_polygon(s, c, stroke=colors[-1]))

"""
def _create_box(layer, dim, angle, colors, pos=(0, 0)):
    x = get_box_points(dim, angle, pos)
    create_box_from_points(layer, x, colors)
"""


def _pt_to_unit(x):
    return x / 2.83464575


def create_text(layer, text_cont, pos, font_size, flip_y=True, rot=None, offset_y=0):
    # Create text element
    text = inkex.etree.SubElement(layer, inkex.addNS('text', 'svg'))
    text.text = str(text_cont)

    if rot is not None:
        text.set("transform", "rotate({})".format(rot))
        pos = rotate_point(pos, rot)

    pos += np.array([0, offset_y])

    # Set text position to center of document.
    text.set('x', str(pos[0]))
    text.set('y', str(pos[1] if not flip_y else -pos[1]))

    # Center text horizontally with CSS style.
    style = {
        'text-align': 'center',
        'text-anchor': 'middle',
        'font-size': str(_pt_to_unit(font_size))
    }

    text.set('style', formatStyle(style))


def create_text_on_line(layer, text, line, font_size, text_offset, rotate=True):
    center = point_center(line)
    direction = line[1] - line[0]
    angle = -np.rad2deg(np.arctan(direction[1] / direction[0])) if direction[0] != 0 else -90

    create_text(layer, text, center, font_size=font_size, offset_y=text_offset, rot=angle if rotate else 0)


'''
def _surround_boxes(self, layer, boxes, is_dec=False):
    """
    32
    01
    """

    label_pad = self.text_offset + self._pt_to_unit(self.font_size)

    pad_s = np.array([self.text_offset] * 2)
    pad_l = self.text_offset + self._pt_to_unit(self.font_size)

    pad_l = np.array([pad_l] * 2)

    center = np.array([
        boxes[-1][4, 0],
        _point_center(boxes[-1][[1, 5], 1])
    ])

    bb_min_l, bb_max_l = self._get_bounds(boxes[0])
    bb_min_r, bb_max_r = self._get_bounds(boxes[-1])

    x0 = bb_min_l
    x2 = bb_max_r
    x1 = np.array([bb_max_r[0], bb_min_r[1]])
    x3 = np.array([bb_min_l[0], bb_max_l[1]])

    x0[0] -= label_pad
    x3[0] -= label_pad

    x1[0] += label_pad
    x2[0] += label_pad

    x2[1] += label_pad
    x3[1] += label_pad

    end_pad = 3 * label_pad

    if not is_dec:
        x0[1] -= end_pad
        x3[1] += end_pad
        text = "Encoder"
    else:
        x1[1] -= end_pad
        x2[1] += end_pad
        text = "Decoder"

    self._create_text_on_line(layer, text, np.array([x3, x2]))

    layer.insert(0, self._create_polygon([x0, x1, x2, x3], "#e1f1e3"))
'''