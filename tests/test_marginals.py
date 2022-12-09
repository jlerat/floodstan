import sys, re, math, json
from io import StringIO
import numpy as np
import pandas as pd
from itertools import product as prod
from scipy.special import gamma
from scipy.stats import pearson3, norm
from pathlib import Path
import pytest
import warnings

from hydrodiy.data.containers import Vector

from nrivfloodfreq import fdist, fsample
from nrivfloodfreqstan import marginals

from test_sample import get_stationids, get_ams

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent

TQDM_DISABLE = True

def test_floodfreqdist(allclose):
    name = "bidule"
    dist = marginals.FloodFreqDistribution(name)

    assert dist.name == name
    assert hasattr(dist, "locn")
    assert hasattr(dist, "logscale")
    assert hasattr(dist, "shape1")

    s = str(dist)
    assert isinstance(s, str)

    dist.locn = 10
    assert dist.locn == 10

    dist.logshape = -1


def test_marginals(allclose):
    stationids = get_stationids()
    distnames = ["GEV", "LogPearson3", "LogNormal", \
                            "Gumbel", "Normal"]
    nparams = 500

    for stationid in stationids:
        streamflow = get_ams(stationid)

        for distname in distnames:
            dist1 = fdist.factory(distname)
            dist2 = marginals.factory(distname)

            # Test lh moments
            p = dist1.fit_lh_moments(streamflow).iloc[0]
            dist1.set_dict_params(p.to_dict())

            # Skip if extreme shape parameter values
            if distname in ["GEV", "LogPearson3"]:
                sh = dist1.kappa if distname == "GEV" else dist1.g
                if sh < marginals.SHAPE_MIN or \
                        sh > marginals.SHAPE_MAX:
                    continue

            dist2.fit_lh_moments(streamflow)
            if distname == "LogNormal":
                assert allclose(dist2.locn, dist1.m, atol=1e-6)
                assert allclose(dist2.logscale, math.log(dist1.s), atol=1e-6)
            elif distname == "Normal":
                assert allclose(dist2.locn, dist1.mu, atol=1e-6)
                assert allclose(dist2.logscale, dist1.logsig, atol=1e-6)
            elif distname == "Gumbel":
                assert allclose(dist2.locn, dist1.tau, atol=1e-6)
                assert allclose(dist2.logscale, dist1.logalpha, atol=1e-6)
            elif distname == "GEV":
                assert allclose(dist2.locn, dist1.tau, atol=1e-6)
                assert allclose(dist2.logscale, dist1.logalpha, atol=1e-6)
                assert allclose(dist2.shape1, dist1.kappa, atol=1e-6)
            elif distname == "LogPearson3":
                assert allclose(dist2.locn, dist1.m, atol=1e-6)
                assert allclose(dist2.logscale, math.log(dist1.s), atol=1e-6)
                assert allclose(dist2.shape1, dist1.g, atol=1e-6)

            params, _ = fsample.bootstrap_lh_moments(dist1, streamflow, nparams)
            desc = f"[{stationid}] Testing {distname}"
            tbar = tqdm(params.iterrows(), total=nparams, \
                        disable=TQDM_DISABLE, desc=desc)
            if TQDM_DISABLE:
                print("\n"+desc)

            for _, param in tbar:
                # Set parameters
                dist1.set_dict_params(param.to_dict())

                # Skip if extreme shape parameter values
                if distname in ["GEV", "LogPearson3"]:
                    sh = dist1.kappa if distname == "GEV" else dist1.g
                    if sh < marginals.SHAPE_MIN or \
                            sh > marginals.SHAPE_MAX:
                        continue

                if distname == "LogNormal":
                    dist2.locn = dist1.m
                    dist2.logscale = math.log(dist1.s)
                elif distname == "Normal":
                    dist2.locn = dist1.mu
                    dist2.logscale = dist1.logsig
                elif distname == "Gumbel":
                    dist2.locn = dist1.tau
                    dist2.logscale = dist1.logalpha
                elif distname == "GEV":
                    dist2.locn = dist1.tau
                    dist2.logscale = dist1.logalpha
                    dist2.shape1 = dist1.kappa
                elif distname == "LogPearson3":
                    dist2.locn = dist1.m
                    dist2.logscale = math.log(dist1.s)
                    dist2.shape1 = dist1.g

                # Compare support, pdf and cdf
                assert allclose(dist1.support, dist2.support)

                pdf1 = dist1.pdf(streamflow)
                pdf2 = dist2.pdf(streamflow)
                assert allclose(pdf1, pdf2)

                cdf1 = dist1.cdf(streamflow)
                cdf2 = dist2.cdf(streamflow)
                assert allclose(cdf1, cdf2)


def test_params_guess(allclose):
    stationids = get_stationids()
    distnames = ["GEV", "LogPearson3", "LogNormal", \
                            "Gumbel", "Normal"]
    nvalues = 1000
    nboot = 50

    for stationid in stationids:
        streamflow = get_ams(stationid)

        for distname in distnames:
            dist = marginals.factory(distname)
            dist.fit_lh_moments(streamflow)

            desc = f"[{stationid}] Testing params guess for {distname}"
            tbar = tqdm(range(nboot), total=nboot, \
                        disable=TQDM_DISABLE, desc=desc)
            if TQDM_DISABLE:
                print("\n"+desc)

            distb = marginals.factory(distname)
            for iboot in tbar:
                ys = dist.rvs(nvalues)
                distb.params_guess(ys)

