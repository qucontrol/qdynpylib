import os
import cmath
from glob import glob

def U_arrow_plot(U, name="U", env='equation*'):
    """
    Take complex 4x4 matrix and visualize it by in LaTeX as a matrix of pointer
    arrows.

    Return snippet of latex code
    """
    lines = []
    if env is not None:
        if env == '$':
            lines.append(r'$')
        else:
            lines.append("\\begin{%s}" % env)
    lines.append(name + r' = \begin{pmatrix}')
    n = U.shape[0]
    for i in xrange(n):
        for j in xrange(n):
            if j==n-1:
                eol = r'\\'
            else:
                eol = r'&'
            r, phi = cmath.polar(U[i,j])
            phi = int(phi * 114.5915590262)
            if (r > 1.0e-3):
                lines.append(
                "\\scalebox{%f}{\\rotatebox{%d}{$\\rightarrow$}} %s"
                % (r, phi, eol))
            else:
                lines.append(eol)
    lines.append(r'\end{pmatrix}')
    if env is not None:
        if env == '$':
            lines.append(r'$')
        else:
            lines.append("\\end{%s}" % env)
    return "\n".join(lines)


def U_arrow_to_png(U_arrow_formula, outfile, keep=False):
    """
    Take the output from U_arrow_plot and convert it to PNG data
    """
    from IPython.display import Image
    outfile = os.path.splitext(outfile)[0]
    with open("%s.tex" % outfile, 'w') as out_fh:
        print >> out_fh, r'\documentclass[preview, border=2pt,convert={outfile=\jobname.png}]{standalone}'
        print >> out_fh, r'\usepackage{amsmath}'
        print >> out_fh, r'\usepackage{graphicx}'
        print >> out_fh, r'\begin{document}'
        print >> out_fh, U_arrow_formula
        print >> out_fh, r'\end{document}'
    os.system("pdflatex %s.tex" % outfile)
    os.system("convert -density 150 %s.pdf -quality 90 %s.png"
              % (outfile, outfile))
    i = Image(filename="%s.png" % outfile)
    if keep:
        for file in glob("%s.*" % outfile):
            os.unlink(file)
    return i


def show_U_arrow(U, name="U"):
    """
    Take complex 4x4 matrix and visualize as a matrix of pointer
    arrows.

    Display in image inside an ipython notebook environment
    """
    from IPython.display import display
    png = U_arrow_to_png(U_arrow_plot(U, name), 'temp_u_arrow.dat')
    display(png)
