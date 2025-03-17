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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gamma as gamma_fun
from scipy.special import gammainc
from scipy.optimize import minimize_scalar, minimize

from hydrodiy.io import csv, iutils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

eps = 1e-250

x_target = 1e-17

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs"
fout.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = Path(__file__).stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

LOGGER.info("Compute gamma incomplete reference values")
res = []
def fun(u, alpha):
    return (eps - gammainc(alpha, u))**2

# Define search space for shapes. More emphasis on small values
def get_shapes(a, n, start, end):
    u = np.linspace(0, 1, n)
    return start + (end - start) * (np.exp(a * u) - 1) / (math.exp(a) - 1)

shapes_a = get_shapes(5, 20, 0.01, 0.58)
shapes_b = get_shapes(5, 1000, 0.001, 2.)
shapes = np.unique(np.concatenate([shapes_a, shapes_b]))

x = np.insert(np.logspace(-9, 9, 10000), 0, 0)
for shape1 in shapes:
    alpha = 4 / shape1 ** 2;

    # Finds starting range
    xx = x.copy()
    for niter in range(5):
        pp = gammainc(alpha, xx)
        i0 = np.where(pp > eps)[0].min()
        x0 = xx[i0]
        x1 = x0 / 100
        xx = np.linspace(x1, x0, len(x))

    g = gammainc(alpha, x0)
    interp = shape1 in shapes_a
    res.append({"x0": x0, "logx0": math.log(x0),
                "shape1": shape1, "alpha": alpha, "gammainc": g,
                "interp": interp})

res = pd.DataFrame(res)
lx0_hat = np.interp(shapes, res.shape1[res.interp], res.logx0[res.interp])
x0_hat = np.exp(lx0_hat)
res.loc[:, "x0_hat"] = x0_hat
res.loc[:, "gammainc_hat"] = gammainc(res.alpha, x0_hat)

# Final values
resi = res.loc[res.interp]
ni = len(resi)
txt1 = f"    row_vector[{ni}] g_threshs = ["
txt2 = f"    row_vector[{ni}] logx0_threshs = ["
for iline, (_, line) in enumerate(resi.iterrows()):
    t = f"{line.shape1:0.8f},"
    txt1 += t if iline < ni - 1 else t[:-1]

    t = f"{line.logx0:0.8f},"
    txt2 += t if iline < ni - 1 else t[:-1]

txt1 += "];"
txt2 += "];"
print(txt1)
print("")
print(txt2)

plt.close("all")
fig, axs = plt.subplots(ncols=2, layout="constrained")
ax = axs[0]
ax.plot(res.shape1, res.x0)
ax.plot(res.shape1, x0_hat)
ax.set(yscale="log")

ax = axs[1]
ax.plot(res.shape1, res.gammainc)
ax.plot(res.shape1, res.gammainc_hat)
ax.set(yscale="log")
plt.show()

LOGGER.info("Process completed")
