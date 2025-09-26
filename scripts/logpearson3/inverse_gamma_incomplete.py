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

eps = 1e-50

x_target = 1e-17

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

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

ninterp = 8
s0 = 0.01
s1 = 1.1755
s2 = 2.
shapes_a = np.insert(get_shapes(5, ninterp + 1, s0, s1), ninterp + 1, s2)
shapes_a_mid = (shapes_a[1:] + shapes_a[:-1])/2
shapes_b = get_shapes(5, 1000, 0.001, 2.)
shapes = np.unique(np.concatenate([shapes_a, shapes_a_mid, shapes_b]))

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
    if shape1 in shapes_a:
        interp = 1
    elif shape1 in shapes_a_mid:
        interp = 2
    else:
        interp = 0

    res.append({"x0": x0, "logx0": math.log(x0),
                "shape1": shape1, "alpha": alpha, "gammainc": g,
                "interp": interp})


res = pd.DataFrame(res)

i1 = res.interp == 1
sbounds = res.loc[i1].shape1.values
res0 = res.loc[i1][:-1]
s0 = res0.shape1.values
f0 = res0.logx0.values
res1 = res.loc[i1][1:]
s1 = res1.shape1.values
f1 = res1.logx0.values
resm = res.loc[res.interp == 2]
sm = resm.shape1.values
fm = resm.logx0.values

a = (2 * f0 + 2 * f1 - 4 * fm) / (s1- s0)**2
b = -2 * s0 * a + (4 * fm - 3 * f0 - f1) / (s1- s0)
c = s0**2 * a - s0 * (4 * fm - 3 * f0 - f1) / (s1- s0) + f0
coefs = {"a": a.round(3), "b": b.round(3), "c": c.round(3)}

slow = sbounds[0]
s = res.shape1
a, b, c = [coefs[l][0] for l in ["a", "b", "c"]]
lx = a * slow**2 + b * slow + c
dlx = 2 * a * slow + b
sbounds = np.insert(sbounds, 0, 0.)
coefs["a"] = np.insert(coefs["a"], 0, 0.)
coefs["b"] = np.insert(coefs["b"], 0, dlx)
coefs["c"] = np.insert(coefs["c"], 0, lx - dlx * slow)

sup = sbounds[-1]
a, b, c = [coefs[l][-1] for l in ["a", "b", "c"]]
lx = a * sup**2 + b * sup + c
dlx = 2 * a * sup + b
n = len(coefs["a"])
sbounds = np.insert(sbounds, n, 10.)
coefs["a"] = np.insert(coefs["a"], n, 0.)
coefs["b"] = np.insert(coefs["b"], n, dlx)
coefs["c"] = np.insert(coefs["c"], n, lx - dlx * sup)

lx0_hat = np.ones_like(res.shape1)
for i in range(len(sbounds)-1):
    slow, sup = sbounds[[i, i+1]]
    ii = (res.shape1 >= slow) & (res.shape1 < sup)
    a, b, c = [coefs[l][i] for l in ["a", "b", "c"]]
    s = res.shape1[ii]
    lx0_hat[ii] = a * s**2 + b * s + c

x0_hat = np.exp(lx0_hat)
res.loc[:, "x0_hat"] = x0_hat
res.loc[:, "gammainc_hat"] = gammainc(res.alpha, x0_hat)

# Final values
n = len(coefs["a"])
txt = {l: f"    row_vector[{n+1 if l == 'g_bounds' else n}] {l} = ["
       for l in ["a", "b", "c", "g_bounds"]}
for i in range(n):
    for nm in txt:
        if nm == "g_bounds":
            t = f"{sbounds[i]:0.3f}"
        else:
            t = f"{coefs[nm][i]:0.3f}"

        t = re.sub("0+$", "0", t)
        t = t + "," if i < n - 1 else t
        if i == 5:
            t += "\n" + " " * 40

        txt[nm] += t

txt["g_bounds"] += f",{sbounds[-1]:0.3f}"

for nm in txt:
    txt[nm] += "];"
    print(txt[nm])

plt.close("all")
fig, axs = plt.subplots(ncols=2, layout="constrained")
ax = axs[0]
ax.plot(res.shape1, res.x0)

i1 = res.interp == 1
ax.plot(res.shape1[i1], res.x0[i1], "or")

ax.plot(res.shape1, x0_hat)
ax.set(yscale="log")

ax = axs[1]
ax.plot(res.shape1, res.gammainc)
ax.plot(res.shape1, res.gammainc_hat)
ax.set(yscale="log")
plt.show()

LOGGER.info("Process completed")
