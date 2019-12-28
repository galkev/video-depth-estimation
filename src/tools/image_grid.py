from PIL import Image
import numpy as np
import os


latex_file = """
\\begingroup%
  \\makeatletter%
  \\providecommand\\color[2][]{{%
    \\errmessage{{(Inkscape) Color is used for the text in Inkscape, but the package 'color.sty' is not loaded}}%
    \\renewcommand\\color[2][]{{}}%
  }}%
  \\providecommand\\transparent[1]{{%
    \\errmessage{{(Inkscape) Transparency is used (non-zero) for the text in Inkscape, but the package 'transparent.sty' is not loaded}}%
    \\renewcommand\\transparent[1]{{}}%
  }}%
  \\providecommand\\rotatebox[2]{{#2}}%
  \\newcommand*\\fsize{{\\dimexpr\\f@size pt\\relax}}%
  \\newcommand*\\lineheight[1]{{\\fontsize{{\\fsize}}{{#1\\fsize}}\\selectfont}}%
  \\ifx\\svgwidth\\undefined%
    \\setlength{{\\unitlength}}{{394.89435151bp}}%
    \\ifx\\svgscale\\undefined%
      \\relax%
    \\else%
      \\setlength{{\\unitlength}}{{\\unitlength * \\real{{\\svgscale}}}}%
    \\fi%
  \\else%
    \\setlength{{\\unitlength}}{{\\svgwidth}}%
  \\fi%
  \\global\\let\\svgwidth\\undefined%
  \\global\\let\\svgscale\\undefined%
  \\makeatother%
  \\begin{{picture}}({size_x},{size_y})%
    \\lineheight{{1}}%
    {content}
    \\setlength\\tabcolsep{{0pt}}%
  \\end{{picture}}%
\\endgroup%
"""


class LatexObject:
    def __init__(self, pos):
        self.pos = pos

    def get_pos(self):
        return np.array(self.pos)

    def set_pos(self, pos):
        self.pos = np.array(pos)

    def get_size(self):
        raise NotImplementedError()

    @staticmethod
    def fix_origin(objs):
        min_pos = min([o.get_pos()[0] for o in objs]), min([o.get_pos()[1] for o in objs])
        print(min_pos)

        for o in objs:
            o.set_pos(o.get_pos() - min_pos)

    def latex(self, offset):
        raise NotImplementedError()


class LatexPictureFile:
    def __init__(self, element):
        self.element = element

    def latex(self, offset):
        size = self.element.get_size()
        return latex_file.format(size_x=size[0], size_y=size[1], content=self.element.latex(offset))

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.latex((0, 0)))


class LatexGrid(LatexObject):
    def __init__(self, children, nrow, pos=(0, 0)):
        super().__init__(pos)

        self.children = children

        num_col = nrow
        num_row = int(np.ceil(len(children) / nrow))

        column_sizes = [0] * num_col
        row_sizes = [0] * num_row

        for i, c in enumerate(children):
            col = i % nrow
            row = i // nrow

            column_sizes[col] = max(column_sizes[col], c.get_size()[0])
            row_sizes[row] = max(row_sizes[row], c.get_size()[1])

        for i, c in enumerate(children):
            col = i % nrow
            row = i // nrow

            c.set_pos([sum(column_sizes[:col]), sum(row_sizes[:row])])

        self.size = sum(column_sizes), sum(row_sizes)

    def get_size(self):
        return self.size

    def latex(self, offset):
        return "\n".join([c.latex(self.pos + offset) for c in self.children])


class LatexGraphics(LatexObject):
    def __init__(self, path, pos=(0, 0)):
        super().__init__(pos)

        self.path = path
        self.size = np.array(Image.open(path).size)
        self.size = self.size / self.size[0]

    def get_size(self):
        return self.size

    def latex(self, offset):
        filename = os.path.basename(self.path)
        return f"\\put({self.pos[0]},{self.pos[1]}){{\\includegraphics[width=\\unitlength,page=1]{{{filename}}}}}"


class LatexGraphicsLabeled(LatexObject):
    def __init__(self, graphics, pos=(0, 0), text_top=None, text_left=None, text_bottom=None, text_right=None):
        super().__init__(pos)

        self.graphics = graphics
        self.graphics.set_pos((0, 0))

        self.texts = [LatexText(t, i*90) if t is not None else None
                      for i, t in enumerate([text_top, text_left, text_bottom, text_right])]

        for i, t in enumerate(self.texts):
            if t is not None:
                if i == 0:
                    pos = self.graphics.get_size()[0] / 2, -t.get_size()[1]

                t.set_pos(pos)

        LatexObject.fix_origin([o for o in ([self.graphics] + self.texts) if o is not None])

    def get_size(self):
        size = self.graphics.get_size()
        print(size)
        for i, t in enumerate(self.texts):
            if t is not None:
                size = size + t.get_size()

        print(size)
        return size

    def latex(self, offset):
        return "\n".join([c.latex(self.pos) for c in ([self.graphics] + self.texts) if c is not None])


class LatexText(LatexObject):
    def __init__(self, text, rotate=0, pos=(0, 0)):
        super().__init__(pos=pos)

        self.text = text
        self.rotate = rotate

    def get_size(self):
        font_height = 0.1

        if abs(self.rotate) != 90:
            return np.array([0, font_height])
        else:
            return np.array([font_height, 0])

    def latex(self, offset):
        pos = self.pos + offset
        return f"\\put({pos[0]}, {pos[1]}){{\\makebox(0,0)[t]{{\\lineheight{{1.25}}"\
               f"\\smash{{\\begin{{tabular}}[t]{{c}}{self.text}\\end{{tabular}}}}}}}}"
