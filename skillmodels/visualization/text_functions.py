"""Collection of functions to generate titles, captions etc.

The functions are used in several parts of skillmodels.

"""
from textwrap import wrap
from os.path import splitext, basename, dirname, join


def decompose_path(path):
    directory = dirname(path)
    file, extension = splitext(basename(path))
    return directory, file, extension


def items_to_text_enumeration(items, name=None):
    if not hasattr(items, "__len__") or isinstance(items, str):
        items = [items]

    assert len(items) >= 1, "Empty list encountered in list_to_text"

    items = [str(i) for i in items]

    if len(items) == 1:
        if name is not None:
            text = "{} {}".format(name, items[0])
        else:
            text = items[0]
    else:
        if name is not None:
            text = name + "s "
        else:
            text = ""
        text += ", ".join(items[:-1])
        text += " and {}".format(items[-1])
    return text


def title_text(basic_name, factors=None, periods=None, stages=None, wrap_width=None):
    assert periods is None or stages is None, "You cannot specify periods and stages."

    assert not (
        periods is None and stages is None
    ), "You have to specify periods or a stages"

    if periods == "all":
        title_p = "All Periods"
    elif periods is None:
        title_p = ""
    else:
        title_p = items_to_text_enumeration(periods, name="Period")

    if factors == "all":
        title_f = "All Factors"
    elif factors is None:
        title_f = ""
    else:
        title_f = items_to_text_enumeration(factors)

    if stages == "all":
        title_s = "All Stages"
    elif stages is None:
        title_s = ""
    else:
        title_s = items_to_text_enumeration(stages, name="Stage")

    title = basic_name
    if factors is not None:
        title += " of {}".format(title_f)
    if periods is not None:
        title += " in {}".format(title_p)
    if stages is not None:
        title += " in {}".format(title_s)

    if wrap_width is not None:
        title = wrap(title, wrap_width)
    return title


def write_figure_tex_snippet(figure_path, title, width=None, height=None):
    if width is None:
        width = 1
    if height is None:
        height = 1
    directory, file, extension = decompose_path(figure_path)
    figure_name = file + extension
    tex_path = join(directory, file + ".tex")

    newline = "\n"
    begin_figure = r"\begin{figure}[h!]\centering" + newline
    end_figure = "\end{figure}" + newline

    include = (
        r"\includegraphics[width={}\textwidth,height={}\textheight,keepaspectratio]{{{}}}"
        + newline
    )
    caption = "\caption{{{}}}" + newline

    with open(tex_path, "w") as t:
        t.write(begin_figure)
        t.write(caption.format(title))
        t.write(include.format(width, height, figure_name))
        t.write(end_figure)
        t.write(newline * 3)


def get_preamble():
    directory = dirname(__file__)
    path = join(directory, "preamble.tex")
    with open(path, "r") as f:
        preamble = f.read()
    return preamble
