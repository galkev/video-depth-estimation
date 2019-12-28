import numpy as np
import colorsys
import copy

from inkscape.helper import *


class InkObject(object):
    def __init__(self, parent, pos):
        self.parent = parent
        # self.pos = None
        self.set_pos(pos)

    def set_pos(self, pos):
         #self.pos = pos
        pass

    """
    def move(self, offset):
        self.pos += offset
        self.create()
    """

    def size_small(self):
        return self.size()

    def size(self):
        b0, b1 = self.bounds()
        return b1 - b0

    def create(self):
        pass

    def bounds(self):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError


class CNNInkObject(InkObject):
    def __init__(self, blocks, parent=None, pos=(0, 0), spacing=None, tight_spacing=False, big_space_idx=None):
        self.blocks = blocks
        self.tight_spacing = tight_spacing
        self.big_space_idx = big_space_idx

        if spacing is None:
            if tight_spacing:
                spacing = 5
            else:
                spacing = 7

        self.spacing = spacing

        parent = self.blocks[0].parent if parent is None else parent

        super(CNNInkObject, self).__init__(parent, pos)

    def get_spacing(self, i, block):
        if self.tight_spacing:
            spacing = block.size_small()[0] + self.spacing
        else:
            if i == self.big_space_idx:
                spacing = 20
            else:
                spacing = self.spacing

            spacing += block.size()[0]

        return spacing

    def set_pos(self, pos):
        pos = np.array(pos)

        for i, block in enumerate(self.blocks):
            block.set_pos(pos)

            pos[0] += self.get_spacing(i, block)

    def size_small(self):
        return self.blocks[-1].points[2] - self.blocks[0].points[0]

    def bounds(self):
        return compute_bounds_union([block.bounds() for block in self.blocks])

    @staticmethod
    def draw_connection(parent, block, block_next, offset_y=0, stroke="black", white_back=False):
        if block_next is not None:
            p1 = point_center(block.points[[1, 5]]) + np.array([0, offset_y])
            p2 = np.array([block_next.bounds()[0, 0], p1[1]])

            if white_back:
                parent.append(create_path([p1, p2], stroke="white", stroke_width=0.5))

            parent.append(create_path([p1, p2], arrow=True, stroke=stroke))

    def draw(self):
        for block, block_next in zip(self.blocks, self.blocks[1:] + [None]):
            block.draw()

            if not self.tight_spacing:
                self.draw_connection(self.parent, block, block_next)


class BoxInkObject(InkObject):
    def __init__(self, points, parent=None, pos=(0, 0), block_type="block"):
        self.points = points
        self.colors = CNNBlockInkObject._colors_3d(CNNBlockInkObject.color_table[block_type])
        self.pos = pos

        super(BoxInkObject, self).__init__(parent, pos)

    def size_small(self):
        return self.points[2] - self.points[0]

    def bounds(self):
        return np.array([self.points[0], self.points[5]])

    def set_pos(self, pos):
        # if self.pos != pos:
        offset = np.array(pos) - self.pos

        self.points += offset

    def scale(self, s):
        """
         65
        324
        01
        """

        w = self.points[1] - self.points[0]
        h = self.points[3] - self.points[0]
        d = self.points[5] - self.points[2]

        self.points[[1, 2, 4, 5]] += (s[0] - 1) * w

        self.points[[2, 3, 5, 6]] += 0.5 * (s[1] - 1) * h
        self.points[[0, 1, 4]] -= 0.5 * (s[1] - 1) * h

        self.points[[4, 5, 6]] += 0.5 * (s[2] - 1) * d
        self.points[[0, 1, 2, 3]] -= 0.5 * (s[2] - 1) * d

    def draw(self):
        create_box_from_points(self.parent, self.points, self.colors)


class PoolNetInkObject(CNNInkObject):
    def __init__(self, blocks_gen, blocks_final, num_nets, pool_layer_indices, pos=(0, 0), *args, **kwargs):
        block_layers = [blocks_gen() for _ in range(num_nets)]

        for i, block_layer in enumerate(block_layers):
            for block in block_layer:
                block.label_w = False
                block.label_h = False
                block.label_n = False  # i == 0

        self.block_layers = list(map(list, zip(*block_layers)))
        self.pool_layer_indices = pool_layer_indices
        self.pool_layers = []

        self.blocks_final = blocks_final

        super(PoolNetInkObject, self).__init__(blocks=None, parent=self.block_layers[0][0].parent,
                                               spacing=5, *args, **kwargs)

        self.set_pos(pos)

    def create_pool_layer(self, pos):
        top_indices = [2, 3, 5, 6]
        bot_indices = [0, 1, 4]

        points = np.zeros_like(self.block_layers[0][0].points)

        points[top_indices] = self.block_layers[0][0].points[top_indices]
        points[bot_indices] = self.block_layers[0][-1].points[bot_indices]

        box = BoxInkObject(points, pos=pos, parent=self.parent, block_type="glob_pool")

        box.scale([1, 1.1, 1])

        return box

    def set_pos(self, pos):
        pos_cur = np.array(pos)

        space_y = self.block_layers[0][0].size()[1] + 20

        for i, block_layer in enumerate(self.block_layers):
            for j, block in enumerate(block_layer):
                block.set_pos(pos_cur + np.array([0, -space_y * j]))

            #pos_cur[0] += self.get_spacing(i, block_layer[0])
            pos_cur[0] = block_layer[0].bounds()[1][0] + self.spacing

            if i in self.pool_layer_indices:
                pool_layer = self.create_pool_layer(pos)
                pool_layer.set_pos(pos_cur)

                self.pool_layers.append(pool_layer)

                # pos_cur[0] += self.get_spacing(-1, pool_layer)
                pos_cur[0] = pool_layer.bounds()[1][0] + self.spacing

        pos_cur[1] += space_y

        for i, block in enumerate(self.blocks_final):
            block.set_pos(pos_cur + np.array([0, -space_y * j]))

            # pos_cur[0] += self.get_spacing(i, block_layer[0])
            pos_cur[0] = block.bounds()[1][0] + self.spacing

    def draw(self):
        pool_layer_idx = 0

        normal_conn_y = 2
        pool_conn_y = -2

        for i, (block_layer, block_layer_next) in enumerate(zip(self.block_layers, self.block_layers[1:] + [None])):
            if i in self.pool_layer_indices:
                pool_layer = self.pool_layers[pool_layer_idx]
                for block in block_layer:
                    self.draw_connection(self.parent, block, pool_layer, offset_y=normal_conn_y)
                pool_layer.draw()
                if block_layer_next is not None:
                    for block, block_next in zip(block_layer, block_layer_next):
                        self.draw_connection(
                            self.parent, pool_layer, block_next,
                            stroke="#8b0000",
                            offset_y=(point_center(block_next.points[[1, 5]]) -
                                      point_center(pool_layer.points[[1, 5]]))[1]
                            + normal_conn_y
                        )

                        self.draw_connection(self.parent, block, block_next,
                                             offset_y=pool_conn_y, white_back=True)
                pool_layer_idx += 1
            else:
                if block_layer_next is not None:
                    for block, block_next in zip(block_layer, block_layer_next):
                        self.draw_connection(self.parent, block, block_next, offset_y=normal_conn_y)

            for block in block_layer:
                block.draw()

        for block, block_next in zip(self.blocks_final, self.blocks_final[1:] + [None]):
            block.draw()

            self.draw_connection(self.parent, block, block_next, offset_y=normal_conn_y)


class GroupInkObject(InkObject):
    def __init__(self, children):
        super(GroupInkObject, self).__init__(None, (0, 0))

        self.children = children

    def draw(self):
        for c in self.children:
            c.draw()


class CNNBlockInkObject(InkObject):
    font_size = 11
    text_offset = 1

    box_angle = 60
    depth_ratio = 0.5
    # num_feat_feat_dim_ratio = 1

    num_feat_decr_factor = 0.2
    feat_dim_decr_factor = 0.2

    color_table = {
        "block": "fdfdff",  # blue
        "block2": "9657b5",  # dark blue
        "layer": "fefffd",  # green
        "up": "fdfffe",  # turquiose
        "conv": "fffdff",  # pink
        "pool": "fffdfd",  # red
        "fc": "0089a2ff",  # dark turquoise
        "rnn": "fffffdff",  # yellow
        "rnn_mod": "fffefd",  # orange
        "do": "9f4e00",  # brown
        "reshape": "6fa759ff",  # grey green

        "conv1": "fffdff",  # pink
        "conv3": "fefffd",  # green
        "conv4": "fffdfd",  # red
        "glob_pool": "fffdfd",  # red
    }

    @staticmethod
    def _rgb2hex(color):
        return '#%02x%02x%02x' % (color[0], color[1], color[2])

    @staticmethod
    def _hex2rgb(hex_code):
        return np.array(tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4)))

    @staticmethod
    def _colors_3d(hex_code):
        c = np.array(colorsys.rgb_to_hls(*CNNBlockInkObject._hex2rgb(hex_code).astype(float) / 0xff)) * 0xff

        # lights = [190, 210, 170, 70]
        lights = [150, 180, 130, 70]

        # lights = [150, 180, 130, 80]

        colors = [np.array([c[0], c[1] * l / 0xff, c[2]]) for l in lights]

        return [CNNBlockInkObject._rgb2hex(np.array(colorsys.hls_to_rgb(*(c / 0xff))) * 0xff) for c in colors]

    def __init__(self, num_feat, feat_dim, parent=None, pos=(0, 0), label_w=True, label_h=True, label_n=True,
                 scale_num_feat=10, scale_feat_dim=60, block_type="block"):
        self.num_feat = num_feat
        self.feat_dim = feat_dim

        self.scale_num_feat = scale_num_feat
        self.scale_feat_dim = scale_feat_dim

        self.colors = CNNBlockInkObject._colors_3d(self.color_table[block_type])

        self.box_dim = [
            self._get_scale(self.num_feat, 192, self.num_feat_decr_factor) * self.scale_num_feat,
            self._get_scale(self.feat_dim, 224, self.feat_dim_decr_factor) * self.scale_feat_dim,
            self._get_scale(self.feat_dim, 224, self.feat_dim_decr_factor) * self.scale_feat_dim * self.depth_ratio
        ]

        if block_type.startswith("block"):
            self.label_w = True
            self.label_h = True
            self.label_n = True
        else:
            self.label_w = False
            self.label_h = False
            self.label_n = False

        # self.label_w = label_w
        # self.label_h = label_h
        # self.label_n = label_n

        self.points = None

        super(CNNBlockInkObject, self).__init__(parent, pos)

    def _get_box_dim(self, num_feat, feat_dim):
        return [
            self._get_scale(num_feat, 192, self.num_feat_decr_factor) * self.scale_num_feat,
            self._get_scale(feat_dim, 224, self.feat_dim_decr_factor) * self.scale_feat_dim,
            self._get_scale(feat_dim, 224, self.feat_dim_decr_factor) * self.scale_feat_dim * self.depth_ratio
        ]

    def set_pos(self, pos):
        """
         65
        324
        01
        """
        self.points = get_box_points(self.box_dim, self.box_angle, pos)

    def _get_scale(self, scale, max_scale, decr_factor):
        s = 1 - decr_factor * np.log2(float(max_scale) / scale)

        if scale <= 3:
            s = max(s, 0.2)
        else:
            s = max(s, 0.1)

        return s
        # ratio = float(scale) / max_scale
        # return decr_factor * ratio + (1 - decr_factor)

    def size_small(self):
        return self.points[2] - self.points[0]

    def bounds(self):
        return np.array([self.points[0], self.points[5]])

    def draw(self):
        group = inkex.etree.SubElement(self.parent, 'g')
        create_box_from_points(group, self.points, self.colors)

        if self.label_n:
            create_text_on_line(group, self.num_feat, self.points[[5, 6]], font_size=self.font_size, text_offset=self.text_offset)

        if self.label_h:
            create_text_on_line(group, self.feat_dim, self.points[[0, 3]], font_size=self.font_size, text_offset=self.text_offset)

        if self.label_w:
            create_text_on_line(group, self.feat_dim, self.points[[3, 6]], font_size=self.font_size, text_offset=self.text_offset)


class CNNCoderInkObject(InkObject):
    def __init__(self, length, side1, side2, parent=None, pos=(0, 0), block_type="block", text=None):
        self.dim = [length,
                    side1, side1 * CNNBlockInkObject.depth_ratio,
                    side2, side2 * CNNBlockInkObject.depth_ratio
                    ]

        self.colors = CNNBlockInkObject._colors_3d(CNNBlockInkObject.color_table[block_type])

        self.points = None
        self.text = text

        super(CNNCoderInkObject, self).__init__(parent, pos)

    def set_pos(self, pos):
        """
         65
        324
        01
        """
        self.points = get_frustum_points(self.dim, CNNBlockInkObject.box_angle, pos)

    def draw(self):
        group = inkex.etree.SubElement(self.parent, 'g')
        create_box_from_points(group, self.points, self.colors)

        if self.text is not None:
            create_text_on_line(group, self.text, [point_center(self.points[[0, 3]]), point_center(self.points[[1, 2]])],
                                font_size=CNNBlockInkObject.font_size, text_offset=0, rotate=False)


    def bounds(self):
        return compute_bounds_union(self.points)


class AEInkObject(InkObject):
    def __init__(self, length, side1, side2, parent=None, pos=(0, 0), block_type="block"):
        self.enc = CNNCoderInkObject(length, side1, side2, parent=parent, pos=pos, block_type=block_type, text="Encoder")
        self.dec = CNNCoderInkObject(length, side2, side1, parent=parent, pos=pos, block_type=block_type, text="Decoder")

        super(AEInkObject, self).__init__(parent, pos)

    def set_pos(self, pos):
        pos = np.array(pos)

        self.enc.set_pos(pos)
        pos[0] += self.enc.size()[0] + 7

        self.dec.set_pos(pos)

    def draw(self):
        self.enc.draw()
        CNNInkObject.draw_connection(self.parent, self.enc, self.dec)
        self.dec.draw()



class SkipConnInkObject(InkObject):
    offset = 5

    def __init__(self, block1, block2, parent=None):
        super(SkipConnInkObject, self).__init__(parent, (0, 0))

        self.block1 = block1
        self.block2 = block2

    def draw(self):
        start = point_center(self.block1.points[[0, 1]])
        end = point_center(self.block2.points[[0, 1]])

        points = [
            start,
            start + np.array([0, -self.offset]),
            [end[0], start[1] + -self.offset],
            end
        ]

        self.parent.append(create_path(points, arrow=True))


def create_ae(parent):
    ae = AEInkObject(30, 25, 8, parent=parent)
    return ae


def _create_block_seq(num_feats, feat_dims, parent, scale_num_feat=10, scale_feat_dim=60, block_types=None):
    blocks = []

    for i, (num_feat, feat_dim) in enumerate(zip(num_feats, feat_dims)):
        block_type = block_types[i] if block_types is not None else "block"

        block = CNNBlockInkObject(num_feat, feat_dim, parent=parent, scale_num_feat=scale_num_feat,
                                  scale_feat_dim=scale_feat_dim, block_type=block_type)
        blocks.append(block)

    return blocks


def create_ae_sample(parent):
    num_feats = [64, 128, 256, 512]
    feat_dims = exp_decreasing_range(224, len(num_feats))

    group_enc, group_dec = inkex.etree.SubElement(parent, 'g'), inkex.etree.SubElement(parent, 'g')

    enc = _create_block_seq(num_feats, feat_dims, group_enc)
    #dec = _create_block_seq(reversed(num_feats), reversed(feat_dims), group_dec)

    enc = CNNInkObject(enc)

    return enc


def create_color_test(parent):
    block_types = ["fc", "rnn", "rnn_mod", "reshape"]

    num_feats = [32] * len(block_types)
    feat_dims = [224] * len(block_types)

    net = CNNInkObject(_create_block_seq(num_feats, feat_dims, parent, block_types=block_types), tight_spacing=True)

    return net


def create_single(parent, num_feat, feat_dim, block_type):
    return CNNBlockInkObject(num_feat, feat_dim, parent, block_type=block_type)


def create_singles(parent, num_feats, feat_dims, block_types, scale_num_feat, pos=(0, 0)):
    return CNNInkObject(
        [CNNBlockInkObject(num_feat=f, feat_dim=d, block_type=t, parent=parent, scale_num_feat=scale_num_feat)
         for f, d, t in zip(num_feats, feat_dims, block_types)],
        tight_spacing=True, pos=pos)


def create_group(parent):
    return inkex.etree.SubElement(parent, 'g')


def create_poolnet(parent):
    depth_ratio_org = CNNBlockInkObject.depth_ratio

    CNNBlockInkObject.depth_ratio = 0.0

    pool_layer_indices = [
        0, 1, 2, 3, 9
    ]

    num_feats_enc = [
        32,
        48,
        64,
        128,
        192,
    ]

    block_types_enc = [
        "conv3",
        "block",  # "conv1", "conv4",
        "block",  # "conv1", "conv4",
        "block",  # "conv1", "conv4",
        "block",  # "conv1", "conv4",
    ]

    feat_dims_enc = [
        256,
        256,
        128,
        64,
        32,
    ]

    num_feats_dec = [
        192,
        192,
        128,
        96,
        48,
    ]

    block_types_dec = [
        "conv3",
        "block",  # "up", "conv1", "conv3",
        "block",  # "up", "conv1", "conv3",
        "block",  # "up", "conv1", "conv3",
        "block",  # "up", "conv1", "conv3",
    ]

    feat_dims_dec = [
        16,
        16,
        32,
        64,
        128,
    ]


    num_feats_final = [
        32, 32, 32,
        1
    ]

    block_types_final = [
        "conv3", "conv3", "conv3",
        "conv1"
    ]

    feat_dims_final = [
        256, 256, 256,
        256
    ]

    cnn_group = create_group(parent)

    scale_num_feat = 5

    blocks_gen = lambda: _create_block_seq(num_feats_enc, feat_dims_enc, cnn_group, scale_num_feat=scale_num_feat,
                                           scale_feat_dim=30, block_types=block_types_enc) + \
                         _create_block_seq(num_feats_dec, feat_dims_dec, cnn_group, scale_num_feat=scale_num_feat,
                                           scale_feat_dim=30, block_types=block_types_dec)

    blocks_final = _create_block_seq(num_feats_final, feat_dims_final, cnn_group, scale_num_feat=scale_num_feat,
                                     scale_feat_dim=30, block_types=block_types_final)

    net = PoolNetInkObject(blocks_gen, blocks_final, num_nets=3,
                           pool_layer_indices=pool_layer_indices, big_space_idx=len(feat_dims_enc) - 1,
                           tight_spacing=True)

    skips = []

    skip_pairs = ((1, 9), (2, 8), (3, 7))

    for s1, s2 in skip_pairs:
        for b1, b2 in zip(net.block_layers[s1], net.block_layers[s2]):
            skips.append(SkipConnInkObject(b1, b2, parent))

    CNNBlockInkObject.depth_ratio = depth_ratio_org

    enc_block1 = create_singles(
        create_group(parent), [num_feats_enc[4]] * 3, [feat_dims_enc[4]] * 3, block_types=["conv1", "conv4"],
        scale_num_feat=3, pos=(0, -100))

    dec_block1 = create_singles(
        create_group(parent), [num_feats_enc[4]] * 3, [feat_dims_enc[4]] * 3, block_types=["up", "conv1", "conv3"],
        scale_num_feat=3, pos=(50, -100))

    return GroupInkObject([net, enc_block1, dec_block1] + skips)


def create_poolnet_ex(parent):
    CNNBlockInkObject.depth_ratio = 0.0

    pool_layer_indices = [
        0, 2, 4, 6, 21
    ]

    num_feats_enc = [
        32,
        48, 48,
        64, 64,
        128, 128,
        192, 192
    ]

    block_types_enc = [
        "conv3",
        "conv1", "conv4",
        "conv1", "conv4",
        "conv1", "conv4",
        "conv1", "conv4",
    ]

    feat_dims_enc = [
        256,
        256, 256,
        128, 128,
        64, 64,
        32, 32
    ]

    num_feats_dec = [
        192,
        192, 192, 192,
        128, 128, 128,
        96, 96, 96,
        48, 48, 48
    ]

    block_types_dec = [
        "conv3",
        "up", "conv1", "conv3",
        "up", "conv1", "conv3",
        "up", "conv1", "conv3",
        "up", "conv1", "conv3",
    ]

    feat_dims_dec = [
        16,
        16, 32, 32,
        32, 64, 64,
        64, 128, 128,
        128, 256, 256
    ]


    num_feats_final = [
        32, 32, 32,
        1
    ]

    block_types_final = [
        "conv3", "conv3", "conv3",
        "conv1"
    ]

    feat_dims_final = [
        256, 256, 256,
        256
    ]

    cnn_group = create_group(parent)

    scale_num_feat = 5

    blocks_gen = lambda: _create_block_seq(num_feats_enc, feat_dims_enc, cnn_group, scale_num_feat=scale_num_feat,
                                           scale_feat_dim=30, block_types=block_types_enc) + \
                         _create_block_seq(num_feats_dec, feat_dims_dec, cnn_group, scale_num_feat=scale_num_feat,
                                           scale_feat_dim=30, block_types=block_types_dec)

    blocks_final = _create_block_seq(num_feats_final, feat_dims_final, cnn_group, scale_num_feat=scale_num_feat,
                                     scale_feat_dim=30, block_types=block_types_final)

    net = PoolNetInkObject(blocks_gen, blocks_final, num_nets=3,
                           pool_layer_indices=pool_layer_indices, big_space_idx=len(feat_dims_enc) - 1,
                           tight_spacing=True)

    return net


def create_poolnet2(parent):
    CNNBlockInkObject.box_angle = 60
    CNNBlockInkObject.depth_ratio = 0.0

    num_feats_enc = [32, 48, 64, 128, 192]
    num_feats_dec = [192, 128, 96, 48, 48]
    num_feats_final_conv = [32, 32, 32]

    feat_dims_enc = [256, 128, 64, 32, 16]
    feat_dims_dec = [56, 104, 200, 392]

    # blocks = blocks[:1]
    cnn_group = create_group(parent)

    blocks_gen = lambda: _create_block_seq(num_feats_enc, feat_dims_enc, cnn_group, scale_num_feat=10, scale_feat_dim=30) + \
                         _create_block_seq(num_feats_dec, feat_dims_dec, cnn_group, scale_feat_dim=30)

    net = PoolNetInkObject(blocks_gen, num_nets=2,
                           pool_layer_indices=[0, 1, 2, 3], big_space_idx=len(feat_dims_enc) - 1)

    #final_block = CNNBlockInkObject(32, )

    return net


def create_recurae(parent):
    num_feats = [32, 48, 64, 128, 192]
    num_feats_enc = num_feats + [num_feats[-1]]
    num_feats_dec = [128, 64, 48, 32, 1]

    feat_dims_enc = [256, 128, 64, 32, 16, 8]
    feat_dims_dec = [16, 32, 64, 128, 256]

    cnn_group = create_group(parent)

    blocks_enc = _create_block_seq(num_feats_enc, feat_dims_enc, cnn_group)
    blocks_dec = _create_block_seq(num_feats_dec, feat_dims_dec, cnn_group)

    skips = [SkipConnInkObject(blocks_enc[i], blocks_dec[-(i+1)], parent=cnn_group) for i in range(5)]

    net = CNNInkObject(blocks_enc + blocks_dec, big_space_idx=len(num_feats_enc) - 1)

    enc_block1 = create_singles(
        create_group(parent), [num_feats_enc[1]] * 4, [feat_dims_enc[2]] * 4, block_types=["layer", "layer", "layer", "pool"],
        scale_num_feat=3, pos=(0, -100))

    dec_block1 = create_singles(
        create_group(parent), [num_feats_enc[1]] * 3, [feat_dims_enc[2]] * 3, block_types=["up", "layer", "layer"],
        scale_num_feat=3, pos=(50, -100))

    return GroupInkObject([net, enc_block1, dec_block1] + skips)


def create_ddff(parent):
    num_feats = [64, 128, 256, 512, 512]
    num_feats_dec = [512, 256, 128, 64, 1]
    feat_dims = [256, 128, 64, 32, 16]
    block_types = ["block"] * 2 + ["block2"] * 3

    cnn_group = create_group(parent)

    blocks_enc = _create_block_seq(num_feats, feat_dims, cnn_group, block_types=block_types)
    blocks_dec = _create_block_seq(num_feats_dec, list(reversed(feat_dims)), cnn_group,
                                   block_types=list(reversed(block_types)))

    skip = SkipConnInkObject(blocks_enc[2], blocks_dec[2], parent=cnn_group)

    net = CNNInkObject(blocks_enc + blocks_dec, big_space_idx=len(feat_dims) - 1)


    enc_block1 = create_singles(
        create_group(parent), [num_feats[1]] * 3, [feat_dims[2]] * 3, block_types=["layer", "layer", "pool"],
        scale_num_feat=3, pos=(0, -100))

    dec_block1 = create_singles(
        create_group(parent), [num_feats[1]] * 3, [feat_dims[2]] * 3, block_types=["up", "layer", "layer"],
        scale_num_feat=3, pos=(50, -100))

    enc_block2 = create_singles(
        create_group(parent), [num_feats[1]] * 5, [feat_dims[2]] * 5,
        block_types=["layer", "layer", "layer", "pool", "do"],
        scale_num_feat=3, pos=(100, -100))

    dec_block2 = create_singles(
        create_group(parent), [num_feats[1]] * 5, [feat_dims[2]] * 5,
        block_types=["up", "layer", "layer", "layer", "do"],
        scale_num_feat=3, pos=(150, -100))

    return GroupInkObject([net, skip, enc_block1, dec_block1, enc_block2, dec_block2])


def create_unet(parent):
    num_feats = [64, 128, 256, 512, 1024]
    feat_dims_enc = [572, 284, 140, 68, 32]
    feat_dims_dec = [56, 104, 200, 392]


    # blocks = blocks[:1]
    cnn_group = create_group(parent)

    blocks_enc = _create_block_seq(num_feats, feat_dims_enc, cnn_group)
    blocks_dec = _create_block_seq(list(reversed(num_feats))[1:], feat_dims_dec, cnn_group)


    skips = [SkipConnInkObject(blocks_enc[i], blocks_dec[-(i+1)], parent=cnn_group) for i in range(4)]

    blocks_dec[-1] = CNNInkObject([blocks_dec[-1]] + [CNNBlockInkObject(2, feat_dims_dec[-1], parent=cnn_group, block_type="conv")],
                                  tight_spacing=True)

    net = CNNInkObject(blocks_enc + blocks_dec, big_space_idx=len(feat_dims_enc) - 1)

    enc_block = create_singles(create_group(parent), [num_feats[2]] * 3, [feat_dims_enc[2]] * 3, block_types=["layer", "layer", "pool"],
                               scale_num_feat=3)

    dec_block = create_singles(create_group(parent), [num_feats[-3]] * 3, [feat_dims_enc[-3]] * 3,
                               block_types=["up", "layer", "layer"],
                               scale_num_feat=3)

    return GroupInkObject([net, enc_block, dec_block] + skips)

"""
def create_unet(parent):
    num_feats = [64, 128, 256, 512, 1024]
    img_dim = [572, 284, 140, 68, 32]

    blocks = []

    for i in range(5):
        if i == 0:
            feat_sizes = [1, num_feats[0], num_feats[0]]
        else:
            feat_sizes = [num_feats[i - 1], num_feats[i], num_feats[i]]

        in_dims = [img_dim[i], img_dim[i] - 2, img_dim[i] - 4]

        sub_blocks = _create_block_seq(feat_sizes, in_dims, parent, scale_num_feat=6)
        block = CNNBlockBlocksInkObject(sub_blocks)

        blocks.append(block)

    # blocks = blocks[:1]

    net = CNNInkObject(blocks, spacing=3)

    net.draw()
"""