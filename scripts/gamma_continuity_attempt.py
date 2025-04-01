#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-03-13 16:52:59.428042
## Comment : Explore continuing log pearson 3 pdf
##
## ------------------------------


import sys
import re
import math
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils


# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = Path(__file__).stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

# Transformation function
def nu(x):
    eps = 1e-30
    return eps**2 / 2 / (np.abs(x) + eps)

#def nu(x):
#    eps = 1e-5
#    e = np.exp(- np.abs(x) / eps)
#    return 2 * e / (1 + e)

def fun(x):
    return (x + np.abs(x)) / 2 + nu(x)

plt.close("all")
mosaic = [["fun"] + ["."] * 2]
aa = [4 / g**2 for g in [1.9, 1.92, 1.95]]
mosaic += [[f"a_{a:0.2f}" for a in aa]]
mosaic += [[f"a_{a:0.2f}_zoom" for a in aa]]
#mosaic += [[f"a_{a:0.2f}_zoom_grad" for a in aa]]
nrows, ncols = len(mosaic), len(mosaic[0])
w, h = 3, 2
fig = plt.figure(figsize=(w * ncols, h * nrows),
                 layout="constrained")
axs = fig.subplot_mosaic(mosaic)

for aname, ax in axs.items():
    if aname == "fun":
        x = np.linspace(-0.5, 1, 1000)
        ax.plot(x, fun(x))
        ax.set(title="Approx fun", xlabel="x", ylabel="approx")
        continue

    a = float(re.sub("a_|_zoom($|.*)", "", aname))

    if re.search("zoom", aname):
        x = np.linspace(-1e-20, 1e-20, 1000)
    else:
        x = np.linspace(-2, 20, 1000)


    yraw = gamma.logpdf(x, a)
    ytrans = gamma.logpdf(fun(x), a)

    y0 = gamma.logpdf(fun(0.), a)
    def objfun(x):
        return (gamma.logpdf(x, a) - y0)**2

    opt = minimize_scalar(objfun, method="brent")
    x0 = opt.x
    cdf0 = gamma.cdf(x0, a)

    ax.plot(x, yraw, lw=4, label="Raw")
    ax.plot(x, ytrans, label="Trans", lw=2)

    putils.line(ax, 1, 0, 0, y0, color="k", ls="--", lw=0.9)
    ylim = ax.get_ylim()
    ax.plot([x0, x0], [ylim[0], y0], "k--", lw=0.9)
    ax.text(x0, ylim[0], f" CDF={cdf0:0.3f}", va="bottom",
            ha="left")
    ax.set_ylim(ylim)

    title = f"a={a:0.2f} (g={2/math.sqrt(a):0.2f})"
    if re.search("zoom", aname):
        title += " (zoom)"
    ylabel = "Grad log pdf" if re.search("grad", aname) else "log pdf"
    ax.set(title=title, xlabel="x", ylabel=ylabel)
    #ax.legend(loc=4, fontsize="x-small")

plt.show()

LOGGER.info("Process completed")
