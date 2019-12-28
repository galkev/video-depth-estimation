#!/usr/bin/env python
# These two lines are only needed if you don't put the script directly into
# the installation directory
import sys
sys.path.append('/snap/inkscape/4693/share/inkscape/extensions')
sys.path.append('/home/kevin/Documents/master-thesis/code/mt-project')

# We will use the inkex module with the predefined Effect base class.
import inkex
# The simplestyle module provides functions for style parsing.
from simplestyle import *

import simpletransform


import numpy as np

from inkscape.helper import *
from inkscape.cnn_helper import *


class MakeCNNEffect(inkex.Effect):
    def __init__(self):
        inkex.Effect.__init__(self)

        self.OptionParser.add_option(
            '-w', '--what', action='store',
            type='string', dest='what', default='World',
            help='What would you like to greet?')

        self.font_size = 11
        self.text_offset = 1

    def _create_layer(self):
        svg = self.document.getroot()

        # Create a new layer.
        layer = inkex.etree.SubElement(svg, 'g')
        layer.set(inkex.addNS('label', 'inkscape'), 'CNN')
        layer.set(inkex.addNS('groupmode', 'inkscape'), 'layer')

        return layer

    def _draw_conn(self, layer, box1, box2, offset=5):
        start = point_center(box1[[0, 1]])
        end = point_center(box2[[0, 1]])

        points = [
            start,
            start + np.array([0, -offset]),
            [end[0], start[1] + -offset],
            end
        ]

        layer.append(create_path(points, arrow=True))

    def effect(self):
        # self._draw_conn(layer, boxes_enc[0], boxes_dec[-1])

        #create_ae_sample(self.current_layer)
        #create_unet(self.current_layer).draw()
        # create_ddff(self.current_layer).draw()

        #create_color_test(self.current_layer).draw()

        # create_ae(self.current_layer).draw()

        # create_poolnet(self.current_layer).draw()

        create_recurae(self.current_layer).draw()


effect = MakeCNNEffect()
effect.affect()
