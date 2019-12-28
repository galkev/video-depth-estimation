import os
import contextlib
from tools.latex_config import configure_mpl_latex
configure_mpl_latex()
import matplotlib.pyplot as plt


dft_path = "/home/kevin/Documents/master-thesis/thesis/figures/plots"
heading_fmt = "textbf"


def latex_format(tag, text):
    if tag is None or tag == "":
        return text
    else:
        return "\\" + tag + "{" + text + "}"


def plot_img(img, pos, width):
    height = width / (img.size[0] / img.size[1])
    plt.imshow(img, extent=[pos[0], width, pos[1], height])


def plot_data(data, show_fig=False):
    for i in range(len(data["x"])):
        x, y = data["x"][i], data["y"][i]
        label = data["label"][i] if "label" in data else None

        plt.plot(x, y, label=label)

    if "label" in data:
        plt.legend()

    if show_fig:
        plt.show()


def plot_labels(xlabel=None, ylabel=None):
    if xlabel is not None:
        plt.xlabel(latex_format(heading_fmt, xlabel))

    if ylabel is not None:
        plt.ylabel(latex_format(heading_fmt, ylabel))


def _init_plot(xlabel=None, ylabel=None, target_size=1, title=None, xlim=None, ylim=None, num_fig=None):
    configure_mpl_latex(target_size)

    fig = plt.figure() # if num_fig is None else

    if title is not None:
        plt.title(latex_format(heading_fmt, title))

    plot_labels(xlabel, ylabel)

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    return fig


def _save_plot(fig, path=dft_path, filename="plot", dpi=None):
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    fig.savefig(os.path.expanduser(os.path.join(path, filename + ".pdf")), bbox_inches='tight', dpi=dpi, pad_inches=0)
    fig.savefig(os.path.expanduser(os.path.join(path, filename + ".pgf")), bbox_inches='tight', dpi=dpi, pad_inches=0)


@contextlib.contextmanager
def plot_latex(xlabel=None, ylabel=None, target_size=1, title=None, xlim=None, ylim=None, path=dft_path,
               filename="plot", dpi=None):
    fig = _init_plot(xlabel, ylabel, target_size, title, xlim, ylim)

    yield

    _save_plot(fig, path, filename, dpi=dpi)
