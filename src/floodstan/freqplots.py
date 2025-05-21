import re
import numpy as np
import pandas as pd

import matplotlib.patheffects as pe

from scipy.stats import norm, lognorm

from hydrodiy.stat import sutils

PLOT_TYPES = ["gumbel", "normal", "lognormal"]
PLOT_TYPE_LABELS = {
    "gumbel": "Gumbel reduced variate -log(-log(F))",
    "normal": "Normal standard deviate",
    "lognormal": "LogNormal standard deviate"
    }


def _check_plot_type(ptype):
    txt = "/".join(PLOT_TYPES)
    if ptype not in PLOT_TYPES:
        errmsg = f"Expected plot type in {txt}, got {ptype}."
        raise ValueError(errmsg)


def cdf_to_reduced_variate(prob, plot_type):
    _check_plot_type(plot_type)
    if plot_type == "gumbel":
        return -np.log(-np.log(prob))
    elif plot_type == "normal":
        return norm.ppf(prob)
    elif plot_type == "lognormal":
        return lognorm.ppf(prob, s=1)


def reduced_variate_to_cdf(x, plot_type):
    _check_plot_type(plot_type)
    if plot_type == "gumbel":
        return np.exp(-np.exp(-x))
    elif plot_type == "normal":
        return norm.cdf(x)
    elif plot_type == "lognormal":
        return lognorm.cdf(x, s=1)


def cdf_to_reduced_variate_equidistant(nval, plot_type, cst=0.4):
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
    return cdf_to_reduced_variate(ppos, plot_type)


def set_xlim(ax, plot_type, ari_min, ari_max):
    x0 = cdf_to_reduced_variate(1 - 1./ari_min, plot_type)
    x1 = cdf_to_reduced_variate(1 - 1./ari_max, plot_type)
    ax.set_xlim((x0, x1))


def set_xlabel(ax, plot_type):
    lab = PLOT_TYPE_LABELS[plot_type]
    ax.set_xlabel(lab)


def xaxis_label(ax, plot_type):
    _check_plot_type(plot_type)
    if plot_type == "gumbel":
        ax.set_xlabel("Gumbel reduced variate [-]")
    elif plot_type == "normal":
        ax.set_xlabel("Standard normal deviate [-]")


def add_aep_to_xaxis(ax, plot_type, full_line=True,
                     return_periods=[5, 10, 50, 100, 200],
                     kwargs_plot=None, kwargs_text=None):
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
    aeps = 1. / np.array(return_periods) * 100
    xpos = cdf_to_reduced_variate(1 - aeps / 100, plot_type)

    kwp = {"color": "gray", "linewidth": 2,
           "linestyle":"-"}
    if kwargs_plot is not None:
        kwp.update(kwargs_plot)
        kwargs_plot = kwp

    kwt = {"color": "gray",
           "va": "bottom", "ha": "center",
           "path_effects":[pe.withStroke(linewidth=3,
                           foreground="w")]}
    if kwargs_text is not None:
        kwt.update(kwargs_text)
        kwargs_text = kwt

    # Handle non-linear axis transforms
    delta = 0.02
    y0, y1 = ax.get_ylim()
    fun = (ax.transAxes + ax.transData.inverted()).transform
    _, y0d1 = fun((0, delta))
    _, y0d2 = fun((0, 2 * delta))

    nextline = "\n"
    for retper, aep, x in zip(return_periods, aeps, xpos):
        ax.plot([x, x], [y0, y0d1], **kwargs_plot)
        aep_txt = re.sub("\\.0+$", "", f"{aep:0.2f}")
        txt = f"{aep_txt}%AEP{nextline}1:{retper:0.0f}Y"
        ax.text(x, y0d2, txt, **kwargs_text)

        if full_line:
            kwp = kwargs_plot.copy()
            kwp["linestyle"] = "--"
            kwp["linewidth"] = 0.5
            ax.plot([x, x], [y0, y1], **kwp)

    ax.set_ylim((y0, y1))

    return aeps, xpos


def plot_data(ax, data, plot_type, **kwargs):
    if data.ndim != 1:
        errmess = "Expected 1-dimensional data."
        raise ValueError(errmess)

    data_sorted = np.sort(data[~np.isnan(data)])
    nval = len(data_sorted)
    rvar = cdf_to_reduced_variate_equidistant(nval, plot_type)

    kwargs["marker"] = kwargs.get("marker", "o")
    kwargs["markerfacecolor"] = kwargs.get("markerfacecolor", "w")
    kwargs["markeredgecolor"] = kwargs.get("markeredgecolor", "k")
    kwargs["color"] = kwargs.get("color", "none")

    ax.plot(rvar, data_sorted, **kwargs)

    return rvar, data_sorted


def aris_to_x(aris, truncated_probability,
              plot_type):
    aris = np.array(aris)
    prob = 1 - 1./aris
    probtrunc = (prob - truncated_probability) / (1 - truncated_probability)

    # Check aris below truncated probability are excluded
    x = np.nan * np.zeros_like(probtrunc)
    iok = probtrunc >= 0
    if iok.sum() == 0:
        errmess = "All aris lead to negative probability"
        raise ValueError(errmess)

    probtrunc[~iok] = np.nan
    x[iok] = cdf_to_reduced_variate(prob[iok], plot_type)

    return x, probtrunc


def aris_to_x_equidistant(Tmin, Tmax, nval, truncated_probability,
                          plot_type):
    x0 = cdf_to_reduced_variate(1. - 1. / Tmin, plot_type)
    x1 = cdf_to_reduced_variate(1. - 1. / Tmax, plot_type)
    xx = np.linspace(x0, x1, nval)
    aris = 1. / (1. - reduced_variate_to_cdf(xx, plot_type))
    x, probtrunc = aris_to_x(aris, truncated_probability, plot_type)
    return aris, x, probtrunc


def set_cdf_as_xticklabels(ax, plot_type, cdfs=None):
    if cdfs is None:
        cdfs = np.array([0.99, 0.9, 0.5, 0.1, 0.01, 0.005])
    x = cdf_to_reduced_variate(cdfs, plot_type)
    ax.set_xticks(x)
    xlabs = [f"{c:0.1e}" for c in cdfs]
    ax.set_xticklabels(xlabs)


def plot_marginal_quantiles(ax, aris, quantiles, plot_type,
                            label="", truncated_probability=0.,
                            color="tab:blue", edgecolor="none",
                            facecolor="none", alpha=0.5, ymin_clip=0.,
                            center_column=None, q0_column="none",
                            q1_column="none",
                            **kwargs):
    # Check inputs
    if not isinstance(quantiles, pd.DataFrame):
        errmess = "Expected quantiles of type pd.DataFrame,"\
                  + f" got {type(quantiles)}."
        raise ValueError(errmess)

    # Detect the column name of central value
    for nm in [str(center_column), "mean", "mle"]:
        if nm in quantiles.columns:
            center_column = nm
    if center_column is None:
        errmess = "No name available for center_column."
        raise ValueError(errmess)

    # Convert aris to prob
    x, probtrunc = aris_to_x(aris, truncated_probability,
                             plot_type)
    kwargs["color"] = kwargs.get("color", color)
    facecolor = color if facecolor == "none" else facecolor

    # Sort values
    k = np.argsort(x)
    x = x[k]
    probtrunc = probtrunc[k]
    quantiles = quantiles.iloc[k]

    # Plot center value
    ym = quantiles.loc[:, center_column].clip(ymin_clip)
    ax.plot(x, ym, label=label, **kwargs)

    # Plot uncertainty band
    if q0_column != "none" and q1_column != "none":
        yq1 = quantiles.loc[:, q0_column].clip(ymin_clip)
        yq2 = quantiles.loc[:, q1_column].clip(ymin_clip)
        ax.fill_between(x, yq1, yq2,
                        edgecolor=edgecolor,
                        facecolor=facecolor,
                        alpha=alpha)

    return x


def plot_marginal_cdf(ax, marginal, plot_type,
                      Tmin=1.1, Tmax=200,
                      label="", truncated_probability=0.,
                      color="tab:blue", facecolor="none",
                      ymin_clip=0., **kwargs):
    aris, x, probtrunc = aris_to_x_equidistant(Tmin, Tmax, 500,
                                               truncated_probability,
                                               plot_type)
    quantiles = marginal.ppf(probtrunc).clip(ymin_clip).squeeze()
    quantiles = pd.DataFrame({"mean": quantiles})
    return plot_marginal_quantiles(ax, aris, quantiles, plot_type,
                                   label, truncated_probability,
                                   color, "none", facecolor,
                                   0., ymin_clip, "mean",
                                   "none", "none", **kwargs)


def plot_marginal_params(ax, marginal, params, plot_type,
                         Tmin=1.1, Tmax=200,
                         label="", coverage=0.9, truncated_probability=0.,
                         color="tab:blue", edgecolor="none",
                         facecolor="none", alpha=0.5, ymin_clip=0.,
                         param_prefix="",
                         **kwargs):
    aris, x, probtrunc = aris_to_x_equidistant(Tmin, Tmax, 500,
                                               truncated_probability,
                                               plot_type)
    ys = []
    for _, p in params.iterrows():
        marginal.locn = p.loc[f"{param_prefix}locn"]
        marginal.logscale = p.loc[f"{param_prefix}logscale"]
        marginal.shape1 = p.loc[f"{param_prefix}shape1"]
        ys.append(marginal.ppf(probtrunc))

    ys = pd.DataFrame(ys).T
    ym = ys.mean(axis=1).clip(ymin_clip).values
    quantiles = pd.DataFrame({"mean": ym})

    if coverage > 0:
        qq = (1 - coverage) / 2
        q0 = ys.quantile(qq, axis=1).clip(ymin_clip).values
        quantiles.loc[:, "q0"] = q0

        q1 = ys.quantile(1 - qq, axis=1).clip(ymin_clip).values
        quantiles.loc[:, "q1"] = q1

    return plot_marginal_quantiles(ax, aris, quantiles, plot_type,
                                   label, truncated_probability,
                                   color, edgecolor, facecolor,
                                   alpha, ymin_clip, "mean",
                                   "q0", "q1", **kwargs)
