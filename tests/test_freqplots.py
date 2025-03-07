import sys, re, math
import numpy as np
import pandas as pd
from itertools import product as prod
from itertools import combinations
from pathlib import Path

import pytest
import warnings

import matplotlib.pyplot as plt

from floodstan import marginals, freqplots

from test_sample_univariate import get_stationids, get_ams

FTESTS = Path(__file__).resolve().parent
FIMG = FTESTS / "images"
FIMG.mkdir(exist_ok=True)

@pytest.mark.parametrize("ptype",
                         freqplots.PLOT_TYPES)
def test_reduced_variate(ptype, allclose):
    nval = 1000
    prob = (np.arange(1, nval+1)-0.4)/(nval+1-0.8)
    rv = freqplots.cdf_to_reduced_variate(prob, ptype)
    prob2 = freqplots.reduced_variate_to_cdf(rv, ptype)
    assert allclose(prob, prob2)

    rv2 = freqplots.cdf_to_reduced_variate(prob2, ptype)
    assert allclose(rv, rv2)

@pytest.mark.parametrize("ptype",
                         freqplots.PLOT_TYPES)
def test_reduced_variate_equidistant(ptype, allclose):
    nval = 4
    data = np.linspace(0, 1, nval)
    rv = freqplots.cdf_to_reduced_variate_equidistant(nval, ptype)
    if ptype == "gumbel":
        expected = np.array([-0.66572981,  0.03554335,  0.73485899,  1.86982471])
    elif ptype == "normal":
        expected = np.array([-1.06757052, -0.30298045,  0.30298045,  1.06757052])
    elif ptype == "lognormal":
        expected = np.array([0.34384286, 0.73861354, 1.35388799, 2.90830525])

    assert allclose(rv, expected)


def test_cdf_to_reduced_variate_gumbel(allclose):
    nval = 20
    rv = freqplots.cdf_to_reduced_variate_equidistant(nval, "gumbel")
    prob = (np.arange(1, nval+1)-0.4)/(nval+1-0.8)
    expected = -np.log(-np.log(prob))
    assert allclose(rv, expected, rtol=0, atol=1e-2)


def test_plot_data():
    streamflow = get_ams("203014")
    for plot_type in freqplots.PLOT_TYPES:
        plt.close("all")
        fig, ax = plt.subplots()
        freqplots.plot_data(ax, streamflow, plot_type)
        freqplots.add_aep_to_xaxis(ax, plot_type)
        fp = FIMG / f"freqlots_data_{plot_type}.png"
        fig.savefig(fp)


@pytest.mark.parametrize("ptype",
                         freqplots.PLOT_TYPES)
def test_plot_marginal_cdf(ptype):
    streamflow = get_ams("203014")

    plt.close("all")
    fig, ax = plt.subplots()
    freqplots.plot_data(ax, streamflow, ptype)

    gev = marginals.GEV()
    for eta in range(4):
        gev.fit_lh_moments(streamflow, eta)
        lab = f"GEV $\eta$={eta}"
        freqplots.plot_marginal_cdf(ax, gev, ptype, label=lab, Tmax=500)

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    ax.legend()
    fp = FIMG/ f"freqplots_marginal_{ptype}.png"
    fig.savefig(fp)


@pytest.mark.parametrize("ptype",
                         freqplots.PLOT_TYPES)
@pytest.mark.parametrize("overtrunc", [True, False])
def test_plot_marginal_cdf_censored(ptype, overtrunc):
    streamflow = get_ams("203014")

    gev = marginals.GEV()
    pcens = 0.4
    cens = streamflow.quantile(pcens)
    icens = streamflow>=cens
    truncated_probability = (~icens).sum()/len(streamflow)

    # fitting model on truncated data
    gev.params_guess(streamflow[icens])

    plt.close("all")
    fig, ax = plt.subplots()
    freqplots.plot_data(ax, streamflow, ptype)

    Tmin = 1./(1.1 - truncated_probability) if overtrunc\
            else 1./(0.9 - truncated_probability)

    freqplots.plot_marginal_cdf(ax, gev, ptype,
                                truncated_probability=truncated_probability,
                                Tmin=Tmin,
                                Tmax=500)
    if overtrunc:
        msg = "All aris lead to"
        with pytest.raises(ValueError, match=msg):
            Tmin = 1./(1.1 - truncated_probability)
            Tmax = 1./(1.05 - truncated_probability)
            freqplots.plot_marginal_cdf(ax, gev, ptype,
                                    truncated_probability=truncated_probability,
                                    Tmin=Tmin,
                                    Tmax=Tmax)


    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    ax.legend()
    fp = FIMG/ f"freqplots_marginal_censored_{ptype}_{overtrunc}.png"
    fig.savefig(fp)


@pytest.mark.parametrize("ptype",
                         freqplots.PLOT_TYPES)
def test_plot_marginal_params(ptype):
    streamflow = get_ams("203014")

    plt.close("all")
    fig, ax = plt.subplots()
    ptype = "gumbel"
    freqplots.plot_data(ax, streamflow, ptype)

    gev = marginals.GEV()
    gev.fit_lh_moments(streamflow)
    p0 = np.array([gev.locn, gev.logscale, gev.shape1])[None, :]
    params = p0+np.abs(p0)*0.1*np.random.normal(size=(100, 3))
    params = pd.DataFrame(params, columns=["locn", "logscale", "shape1"])

    freqplots.plot_marginal_params(ax, gev, params, ptype, label="GEV",
                                   Tmax=500,
                                   coverage=0.99,
                                   facecolor="tab:pink",
                                   edgecolor="k",
                                   alpha=0.3)

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    ax.legend()
    fp = FIMG/ f"freqplots_marginal_params_{ptype}.png"
    fig.savefig(fp)


@pytest.mark.parametrize("ptype",
                         freqplots.PLOT_TYPES)
@pytest.mark.parametrize("uncertainty", [True, False])
def test_plot_marginal_quantiles(ptype, uncertainty):
    streamflow = get_ams("203014")

    plt.close("all")
    fig, ax = plt.subplots()
    freqplots.plot_data(ax, streamflow, ptype)

    # Generate data
    gev = marginals.GEV()
    gev.fit_lh_moments(streamflow)
    p0 = np.array([gev.locn, gev.logscale, gev.shape1])[None, :]
    params = p0+np.abs(p0)*0.1*np.random.normal(size=(100, 3))
    params = pd.DataFrame(params, columns=["locn", "logscale", "shape1"])

    aris = np.array([2, 5, 10, 20, 50, 100, 500, 1000])
    probtrunc = 1 - 1./aris
    ys = []
    for _, p in params.iterrows():
        gev.locn = p.locn
        gev.logscale = p.logscale
        gev.shape1 = p.shape1
        ys.append(gev.ppf(probtrunc))

    ys = pd.DataFrame(ys).T
    ym = ys.mean(axis=1).values
    quantiles = pd.DataFrame({"mean": ym})

    coverage = 0.9
    qq = (1 - coverage) / 2
    q0 = ys.quantile(qq, axis=1).values
    quantiles.loc[:, "q0"] = q0

    q1 = ys.quantile(1 - qq, axis=1).values
    quantiles.loc[:, "q1"] = q1

    if uncertainty:
        freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                      q0_column="q0",
                                      q1_column="q1",
                                      label="GEV 2",
                                      alpha=0.3,
                                      facecolor="tab:blue",
                                      edgecolor="k")
    else:
        freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                      label="GEV", color="red")

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    ax.legend()
    fp = FIMG/ f"freqplots_marginal_quantiles_{ptype}_uncertainty{uncertainty}.png"
    fig.savefig(fp)
