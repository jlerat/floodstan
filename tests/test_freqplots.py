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

from test_sample import get_stationids, get_ams

FTESTS = Path(__file__).resolve().parent
FIMG = FTESTS / "images"
FIMG.mkdir(exist_ok=True)

def test_variate(allclose):
    nval = 4
    data = np.linspace(0, 1, nval)
    for plot_type in freqplots.PLOT_TYPES:
        rv = freqplots.reduced_variate(nval, plot_type)
        if plot_type == "gumbel":
            expected = np.array([-0.66572981,  0.03554335,  0.73485899,  1.86982471])
        elif plot_type == "normal":
            expected = np.array([-1.06757052, -0.30298045,  0.30298045,  1.06757052])

        assert allclose(rv, expected)


def test_reduced_variate_gumbel(allclose):
    nval = 20
    rv = freqplots.reduced_variate(nval, "gumbel")
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


def test_plot_marginal():
    streamflow = get_ams("203014")

    plt.close("all")
    fig, ax = plt.subplots()
    ptype = "gumbel"
    freqplots.plot_data(ax, streamflow, ptype)

    gev = marginals.GEV()
    for eta in range(4):
        gev.fit_lh_moments(streamflow, eta)
        lab = f"GEV $\eta$={eta}"
        freqplots.plot_marginal(ax, gev, ptype, label=lab, Tmax=500)

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    ax.legend()
    fp = FIMG/ "freqplots_marginal.png"
    fig.savefig(fp)


def test_plot_marginal_censored():
    streamflow = get_ams("203014")

    gev = marginals.GEV()
    cens = streamflow.median()
    icens = streamflow>=cens
    pcensored = (~icens).sum()/len(streamflow)

    gev.params_guess(streamflow[icens])

    plt.close("all")
    fig, ax = plt.subplots()
    ptype = "gumbel"
    freqplots.plot_data(ax, streamflow, ptype)

    freqplots.plot_marginal(ax, gev, ptype, \
                                pcensored=pcensored, Tmax=500)

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    ax.legend()
    fp = FIMG/ "freqplots_marginal_censored.png"
    fig.savefig(fp)



def test_plot_marginal_uncertainty():
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

    freqplots.plot_marginal(ax, gev, ptype, params=params, label="GEV", \
                        Tmax=500, edgecolor="tab:green")

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    ax.legend()
    fp = FIMG/ "freqplots_marginal_uncertainty.png"
    fig.savefig(fp)


