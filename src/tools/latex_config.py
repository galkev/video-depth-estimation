import matplotlib as mpl
import numpy as np


def figsize(scale):
    # fig_width_pt = 469.755
    fig_width_pt = 418.25555                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    if isinstance(scale, list):
        fig_width = fig_width_pt * inches_per_pt * scale[0]  # width in inches
        fig_height = fig_width_pt * inches_per_pt * scale[1]  # height in inches
    else:
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


# size relative to linewith in [0, 1]
def configure_mpl_latex(size=0.8):
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 11,  # LaTeX default is 10pt font.
        "font.size": 11,
        "legend.fontsize": 9,  # Make the legend/label fonts a little smaller
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": figsize(size),  # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[sc]{mathpazo}"
            r"\linespread{1.05}"
            r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        ]
    }
    mpl.rcParams.update(pgf_with_latex)
