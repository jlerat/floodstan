import math, re
from itertools import product as prod
import numpy as np
import pandas as pd

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from hydrodiy.stat import sutils

PLOT_TYPES = ["gumbel", "normal"]

def _check_plot_type(ptype):
    txt = "/".join(PLOT_TYPES)
    errmsg = f"Expected plot type in {txt}, got {ptype}."
    assert ptype in PLOT_TYPES, errmsg


def get_quantiles(prob, plot_type):
    _check_plot_type(plot_type)
    if plot_type == "gumbel":
        return -np.log(-np.log(prob))
    elif plot_type == "normal":
        return norm.ppf(prob)


def reduced_variate(nval, plot_type, cst=0.4):
    """Generate reduced variates.

    Parameters
    ----------
    nval : int
        Number of points
    plot_type : str
        Plot type. See fplots.PLOT_TYPES.
    cst : float
        Parameter controlling the plotting position.
        Set to 0.4 to use Cunnane's method (Cunnane, 1978).

    Returns
    -------
    rvar : np.ndarray
        1D array containing reduced variates.
    """
    ppos = sutils.ppos(nval, cst=0.4)
    return get_quantiles(ppos, plot_type)


def xaxis_label(ax, plot_type):
    _check_plot_type(plot_type)
    if plot_type == "gumbel":
        ax.set_xlabel("Gumbel reduced variate [-]")
    elif plot_type == "normal":
        ax.set_xlabel("Standard normal deviate [-]")


def add_aep_to_xaxis(ax, plot_type, full_line=True, \
                        return_periods=[5, 10, 50, 100, 200], \
                        kwargs_plot={"color": "gray", "linewidth": 2}, \
                        kwargs_text={"color": "gray", "va": "bottom", "ha": "center"}):
    """ Add annual exceedance probabilities (AEP) to x axis.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axe to draw on.
    plot_type : str
        Plot type. See fplots.PLOT_TYPES.
    return_periods : list
        List of reference return periods to plot.
    """

    aeps = 1/np.array(return_periods)*100
    xpos = get_quantiles(1-aeps/100, plot_type)

    # Handle non-linear axis transforms
    delta = 0.02
    y0, y1 = ax.get_ylim()
    fun = (ax.transAxes + ax.transData.inverted()).transform
    _, y0d1 = fun((0, delta))
    _, y0d2 = fun((0, 2*delta))

    nextline = "\n"
    for retper, aep, x in zip(return_periods, aeps, xpos):
        ax.plot([x, x], [y0, y0d1], **kwargs_plot)
        aep_txt = re.sub("\.0+$", "", f"{aep:0.2f}")
        txt = f"{aep_txt}%{nextline}{retper:0.0f}Y"
        ax.text(x, y0d2, txt, **kwargs_text)

        if full_line:
            kwargs_plot_full = kwargs_plot.copy()
            kwargs_plot_full["linewidth"] = kwargs_plot_full.get("linewidth", 2)/4
            kwargs_plot_full["linestyle"] = kwargs_plot_full.get("linestyle", "--")
            ax.plot([x, x], [y0, y1], **kwargs_plot_full)

    ax.set_ylim((y0, y1))

    return aeps, xpos


def plot_data(ax, data, plot_type, **kwargs):
    assert data.ndim == 1
    data_sorted = np.sort(data)
    nval = len(data)

    rvar = reduced_variate(nval, plot_type)

    kwargs["marker"] = kwargs.get("marker", "o")
    kwargs["markerfacecolor"] = kwargs.get("markerfacecolor", "w")
    kwargs["markeredgecolor"] = kwargs.get("markeredgecolor", "k")
    kwargs["color"] = kwargs.get("color", "none")

    ax.plot(rvar, data_sorted, **kwargs)

    return rvar, data_sorted


def plot_marginal(ax, marginal, plot_type, params=None, Tmin=1.1, Tmax=200, \
                        label="", coverage=0.9, truncated_probability=0., \
                        color="tab:blue", edgecolor="none", \
                        facecolor="none", alpha=0.5, y_min_clip=0., \
                        **kwargs):
    # Compute probabilitie including censoring shift
    prob = np.linspace(1-1/Tmin, 1-1/Tmax, 500)
    probtrunc = (prob-truncated_probability)/(1-truncated_probability)
    x = get_quantiles(prob, plot_type)

    if not params is None:
        assert len(params)>0, "0 parameter sets provided"

    # Colors
    kwargs["color"] = kwargs.get("color", color)
    facecolor = color if facecolor=="none" else facecolor

    if params is None:
        ys = marginal.ppf(probtrunc)
        ax.plot(x, ys, label=label, **kwargs)
    else:
        ys = []
        for _, p in params.iterrows():
            marginal.locn = p.locn
            marginal.logscale = p.logscale
            marginal.shape1 = p.shape1
            ys.append(marginal.ppf(probtrunc))

        ys = pd.DataFrame(ys).T
        ym = ys.mean(axis=1).clip(y_min_clip)
        ax.plot(x, ym, label=label, **kwargs)

        if coverage>0:
            qq = (1-coverage)/2
            yq1 = ys.quantile(qq, axis=1).clip(y_min_clip)
            yq2 = ys.quantile(1-qq, axis=1).clip(y_min_clip)

            ax.fill_between(x, yq1, yq2, \
                                edgecolor=edgecolor, \
                                facecolor=facecolor, \
                                alpha=alpha)
    return x, ys


