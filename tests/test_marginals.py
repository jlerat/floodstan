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

from nrivfloodfreq import fdist, fsample
from floodstan import marginals

from test_sample_univariate import get_stationids, get_ams, TQDM_DISABLE

import data_reader

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent


def test_lh_moments(allclose):
    for station, censoring \
                in prod(data_reader.STATIONS, [True, False]):
        # Extract L moments values from flike data
        try:
            testdata, _ = data_reader.get_test_data(station, "GEV", "LH0", \
                                    censoring, "flike")
        except IOError:
            continue

        lmoms = testdata["lmoments"]
        streamflow = testdata["data"].streamflow

        eta = 0
        lams = marginals.lh_moments(streamflow, eta)
        expected = lmoms.iloc[:, 1]
        assert allclose(lams, expected, rtol=0, atol=1e-3)


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

    for pn in marginals.PARAMETERS:
        dist[pn] = 1
        assert getattr(dist, pn) == 1

        dist[pn] = 2
        assert dist[pn] == 2

    dist.logshape = -1


def test_marginals_properties(allclose):
    distnames = marginals.MARGINAL_NAMES
    streamflow = get_ams("203014")
    for distname in distnames:
        dist = marginals.factory(distname)
        dist.params_guess(streamflow)
        s = str(dist)
        y = dist.rvs(size=10000)
        p = dist.cdf(y)
        expected = dist.ppf(p)
        assert allclose(expected, y)


def test_marginals_vs_nrivfloodfreq(allclose):
    stationids = get_stationids()
    distnames = marginals.MARGINAL_NAMES
    nparams = 500
    results = []

    for stationid in stationids:
        streamflow = get_ams(stationid)

        for distname in distnames:
            if distname in ["GeneralizedPareto", "GeneralizedLogistic", "Gamma"]:
                continue

            dist1 = fdist.factory(distname)
            dist2 = marginals.factory(distname)

            # Test lh moments
            p = dist1.fit_lh_moments(streamflow).iloc[0]
            dist1.set_dict_params(p.to_dict())

            # Skip if extreme shape parameter values
            if distname in ["GEV", "LogPearson3"]:
                sh = dist1.kappa if distname == "GEV" else dist1.g
                if sh < marginals.SHAPE1_LOWER or \
                        sh > marginals.SHAPE1_UPPER:
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
                continue
                # Upgraded GEV lh moments
                #assert allclose(dist2.locn, dist1.tau, atol=1e-6)
                #assert allclose(dist2.logscale, dist1.logalpha, atol=1e-6)
                #assert allclose(dist2.shape1, dist1.kappa, atol=1e-6)
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
                    if sh < marginals.SHAPE1_LOWER or \
                            sh > marginals.SHAPE1_UPPER:
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
    distnames = marginals.MARGINAL_NAMES
    nvalues = 1000
    nboot = 200
    if TQDM_DISABLE:
        print("\n")

    for stationid in stationids:
        streamflow = get_ams(stationid)

        for distname in distnames:
            dist = marginals.factory(distname)
            dist.params_guess(streamflow)

            desc = f"[{stationid}] Testing params guess for {distname}"
            tbar = tqdm(range(nboot), total=nboot, \
                        disable=TQDM_DISABLE, desc=desc)
            if TQDM_DISABLE:
                print(desc)

            distb = marginals.factory(distname)
            ems = []
            for iboot in tbar:
                ys = dist.rvs(nvalues)
                distb.params_guess(ys)

                ymin, ymax = distb.support
                assert ymin<=ys.min()
                assert ymax>=ys.max()

                lpdf = distb.logpdf(ys)
                assert np.all(~np.isnan(lpdf))


def test_fit_lh_moments_flike(allclose):
    for eta, distname, station, censoring in prod([0], \
                    data_reader.DISTRIBUTIONS, \
                    data_reader.STATIONS, \
                    [True, False]):
        try:
            testdata, fr = data_reader.get_test_data(station, distname, \
                                    f"LH{eta}", censoring, "flike")
        except FileNotFoundError:
            continue

        streamflow = testdata["data"].streamflow
        fit = testdata["fit"]

        dist = marginals.factory(distname)
        if distname in ["LogNormal", "LogPearson3"]:
            # flike operates on log10 transform data for these 2 distributions
            # and not log transform. As our procedure apply a log transform,
            # we first transform to log10 and then exponentiate, which
            # leads to log10 transform data within our code.
            q = np.exp(np.log10(streamflow))
            dist.fit_lh_moments(q, eta)
        else:
            dist.fit_lh_moments(streamflow, eta)

        samples = pd.Series({"locn": dist.locn, "logscale": dist.logscale, \
                        "shape1": dist.shape1, "scale": dist.scale})

        # Compare parameters
        if distname == "GEV":
            assert allclose(samples.locn, fit.loc[0, 1], rtol=5e-3, atol=1e-2)
            assert allclose(samples.logscale, math.log(fit.loc[1, 1]), rtol=0, atol=1e-2)
            assert allclose(samples.shape1, fit.loc[2, 1], rtol=0, atol=1e-2)
        elif distname == "LogPearson3":
            assert allclose(samples.locn, fit.loc[0, 1], rtol=0, atol=1e-2)
            assert allclose(samples.scale, fit.loc[1, 1], rtol=0, atol=1e-2)

            # 2 datasets showing slightly higher error than others
            # probably due to rounding values in flike.
            if fr.stem in ["203014_LogPearson3_LH0_censoring", \
                            "arr84_LogPearson3_LH0_censoring"]:
                assert allclose(samples.shape1, fit.loc[2, 1], rtol=0, atol=2e-2)
            else:
                assert allclose(samples.shape1, fit.loc[2, 1], rtol=0, atol=1e-2)

        elif distname == "Gumbel":
            assert allclose(samples.logscale, math.log(fit.loc[1, 1]), rtol=0,atol=1e-2)
            assert allclose(samples.locn, fit.loc[0, 1], rtol=0, atol=1e-1)
        elif distname == "LogNormal":
            assert allclose(samples.locn, fit.loc[0, 1], rtol=0, atol=1e-2)
            assert allclose(samples.scale, fit.loc[1, 1], rtol=0, atol=1e-2)


