import math
import re
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from scipy.special import polygamma

from nrivfloodfreq import fdist, fsample
from floodstan import marginals

from test_sample_univariate import get_stationids, get_ams

import data_reader

SEED = 5446

FTESTS = Path(__file__).resolve().parent


@pytest.mark.parametrize("station",
                         data_reader.STATIONS)
@pytest.mark.parametrize("censoring", [True, False])
def test_lh_moments(station, censoring, allclose):
    # Extract L moments values from flike data
    try:
        testdata, _ = data_reader.get_test_data(station, "GEV", "LH0", \
                                censoring, "flike")
    except IOError:
        pytest.skip("No test data")

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
    assert hasattr(dist, "shape1_prior_loc")
    assert hasattr(dist, "shape1_prior_scale")

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

    pp = [
        [10, 1, 0.1],
        np.array([10, 1, 0.1]),
        pd.Series([10, 1, 0.1]),
        ]
    for p in pp:
        dist.params = p
        assert allclose(dist.locn, 10)
        assert allclose(dist.logscale, 1)
        assert allclose(dist.shape1, 0.1)

    with pytest.raises(NotImplementedError,
                       match="Method params_guess"):
        dist.params_guess(np.random.uniform(-1, 1, 10))

    with pytest.raises(ValueError, match="Expected 3 parameters"):
        dist.params = [10, 1]

    with pytest.raises(ValueError, match="Invalid"):
        dist.params = [10, 1, 1000]

    with pytest.raises(ValueError, match="Invalid"):
        dist.params = [10, 1, np.nan]

    with pytest.raises(ValueError, match="Invalid value for"):
        dist.shape1_prior_scale = 0


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
def test_marginals_properties(distname, allclose):
    streamflow = get_ams("203014")
    dist = marginals.factory(distname)
    dist.params_guess(streamflow)
    s = str(dist)
    y = dist.rvs(size=10000)
    p = dist.cdf(y)
    expected = dist.ppf(p)
    assert allclose(expected, y)


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         get_stationids())
def test_marginals_vs_nrivfloodfreq(distname, stationid, allclose):
    nparams = 500
    streamflow = get_ams(stationid)

    if distname in ["GeneralizedPareto", "GeneralizedLogistic", "Gamma"]:
        pytest.skip(f"Skipping {distname}")

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
            pytest.skip("Skipping test because shape is out of bounds")

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
        pass
        # Upgraded GEV lh moments
        #assert allclose(dist2.locn, dist1.tau, atol=1e-6)
        #assert allclose(dist2.logscale, dist1.logalpha, atol=1e-6)
        #assert allclose(dist2.shape1, dist1.kappa, atol=1e-6)

    elif distname == "LogPearson3":
        assert allclose(dist2.locn, dist1.m, atol=1e-6)
        assert allclose(dist2.logscale, math.log(dist1.s), atol=1e-6)
        assert allclose(dist2.shape1, dist1.g, atol=1e-6)

    params, _ = fsample.bootstrap_lh_moments(dist1, streamflow, nparams)
    for _, param in params.iterrows():
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


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         get_stationids())
def test_params_guess(distname, stationid, allclose):
    nvalues = 1000
    nboot = 200
    streamflow = get_ams(stationid)

    dist = marginals.factory(distname)
    dist.params_guess(streamflow)

    distb = marginals.factory(distname)
    for iboot in range(nboot):
        ys = dist.rvs(nvalues)
        distb.params_guess(ys)

        ymin, ymax = distb.support
        assert ymin <= ys.min()
        assert ymax >= ys.max()

        lpdf = distb.logpdf(ys)
        assert np.all(~np.isnan(lpdf))


@pytest.mark.parametrize("distname",
                         data_reader.DISTRIBUTIONS)
@pytest.mark.parametrize("station",
                         data_reader.STATIONS)
def test_quantile_vs_flike(distname, station, allclose):
    try:
        testdata, fr = data_reader.get_test_data(station, distname,
                                                 "LH0", False, "flike")
    except FileNotFoundError:
        pytest.skip("No test data.")

    qt = testdata["quantiles"].loc[:, "quantile"]
    pp = 1-1/qt.index

    dist = marginals.factory(distname)

    # Set parameters
    fit = testdata["fit"]

    if distname == "GEV":
        dist.locn = fit.loc[0, 1]
        dist.logscale = math.log(fit.loc[1, 1])
        dist.shape1 = fit.loc[2, 1]

    elif distname == "LogPearson3":
        dist.locn = fit.loc[0, 1]
        dist.logscale = math.log(fit.loc[1, 1])
        dist.shape1 = fit.loc[2, 1]

    elif distname == "Gumbel":
        dist.locn = fit.loc[0, 1]
        dist.logscale = math.log(fit.loc[1, 1])

    elif distname == "LogNormal":
        dist.locn = fit.loc[0, 1]
        dist.logscale = math.log(fit.loc[1, 1])

    qt2 = dist.ppf(pp)
    if distname in ["LogNormal", "LogPearson3"]:
        # flike operates on log10 transform data for these 2 distributions
        # and not log transform. As our procedure apply a log transform,
        # we first transform to log10 and then exponentiate, which
        # leads to log10 transform data within our code.
        qt2 = 10**np.log(qt2)

    assert allclose(qt, qt2, atol=0.1, rtol=5e-3)


@pytest.mark.parametrize("distname",
                         data_reader.DISTRIBUTIONS)
@pytest.mark.parametrize("station",
                         data_reader.STATIONS)
@pytest.mark.parametrize("censoring", [True, False])
def test_fit_lh_moments_flike(distname, station, censoring, allclose):
    eta = 0
    try:
        testdata, fr = data_reader.get_test_data(station, distname, \
                                f"LH{eta}", censoring, "flike")
    except FileNotFoundError:
        pytest.skip("No test data.")

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


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid", get_stationids()[:3])
@pytest.mark.parametrize("censoring", [False, True])
def test_marginals_maxpost_numerical(distname, stationid, censoring, allclose):
    streamflow = get_ams(stationid)
    marginal = marginals.factory(distname)
    censor = streamflow.quantile(0.33) if censoring else -1e10
    lmp, theta_lmp, dcens, ncens, cov = \
        marginal.maximum_posterior_estimate(streamflow,
        censor, nexplore=10000, explore_scale=0.3)

    # Sample params from laplace approx
    params = np.random.multivariate_normal(mean=theta_lmp,
                                           cov=cov, size=5000)

    # Perturb lmp param and check lp_mlp is always greater
    # with a small tolerance
    #trans_lmp = np.arcsinh(theta_lmp)
    #perturb = np.random.normal(loc=0., scale=0.2,
    #                           size=(5000, 3))
    #for pert in perturb:
    #    theta = np.sinh(trans_lmp + pert)

    for theta in params:
        lp = -marginal.neglogpost(theta, dcens, censor, ncens)
        if np.isfinite(lp) and not np.isnan(lp):
            mess = f"[{distname}/{stationid}/{censoring}] "\
                   + f" num: lp={lp:0.1e} theta={theta} "\
                   + f" lmp: lp={lmp:0.1e} theta={theta_lmp}"
            eps = 5e-2
            assert lp < lmp + eps, print(mess)


@pytest.mark.parametrize("distname", ["Gamma", "LogNormal", "Normal"])
@pytest.mark.parametrize("stationid", get_stationids())
def test_marginals_mle_theoretical(distname, stationid, allclose):
    streamflow = get_ams(stationid)
    marginal = marginals.factory(distname)

    # Set very wide prior scale to get max likelihood
    marginals.SHAPE1_PRIOR_SCALE_MAX = 1e100
    if marginal.has_shape:
        marginal.shape1_prior_scale = 1e100

    mlp_mle, theta_mle, dcens, ncens, cov = \
        marginal.maximum_posterior_estimate(streamflow,
                                            nexplore=50000)
    # Theoretical value of MLE
    if distname == "LogNormal":
        marginal.params_guess(streamflow)
        locn, logscale = marginal.params[:2]

        # Check lognorm mle
        locn0 = np.log(streamflow).mean()
        logscale0 = math.log(math.sqrt(((np.log(streamflow)-locn0)**2).mean()))
        assert allclose(locn, locn0, atol=1e-8, rtol=0.)
        assert allclose(logscale, logscale0, atol=1e-8, rtol=0.)

    elif distname == "Normal":
        marginal.params_guess(streamflow)
        locn, logscale = marginal.params[:2]

        # Check norm mle
        locn0 = streamflow.mean()
        logscale0 = math.log(math.sqrt(((streamflow-locn0)**2).mean()))
        assert allclose(locn, locn0, atol=1e-8, rtol=0.)
        assert allclose(logscale, logscale0, atol=1e-8, rtol=0.)

    elif distname == "Gamma":
        marginal.params_guess(streamflow)
        locn, logscale = marginal.params[:2]

    assert allclose(theta_mle[0], locn, atol=1e-5, rtol=1e-5)
    assert allclose(theta_mle[1], logscale, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         data_reader.STATIONS_BESTFIT)
@pytest.mark.parametrize("censoring", [False, True])
def test_marginals_vs_bestfit(distname, stationid, censoring, allclose):
    if distname == "LogNormal":
        pytest.skip("Problem with LogNormal in bestfit")
    streamflow = get_ams(stationid)

    bestfit, censor = data_reader.read_bestfit_mle(stationid, censoring)
    bestfit = bestfit.loc[:, distname]

    marginal = marginals.factory(distname)

    theta_bestfit = np.array([float(bestfit["Location"]),
                              math.log(float(bestfit["Scale"])),
                              float(bestfit["Shape"])])

    # Take into account bestfit parameterisation
    if distname in ["LogPearson3"]:
        theta_bestfit[0] = math.log(10**theta_bestfit[0])
        theta_bestfit[1] = math.log(math.log(10**float(bestfit["Scale"])))

    elif distname == "LogNormal":
        mean, std = bestfit.iloc[:2].astype(float)
        mu = math.log(mean / math.sqrt(1 + std**2 / mean**2))
        sig = math.log(1 + std**2 / mean**2)
        theta_bestfit[0] = mu
        theta_bestfit[1] = math.log(sig)

    elif distname == "Gamma":
        th, sc = theta_bestfit[:2]
        sc = math.exp(sc)
        theta_bestfit[0] = th*sc
        theta_bestfit[1] = math.log(th)

    # Fix 2 param distributions
    if re.search("^(LogN|Norm|Gum|Gam)", distname):
        theta_bestfit[-1] = 0

    marginal.params = theta_bestfit

    # Check quantiles
    se = bestfit.iloc[16:-3]
    cdfs = 1. - se.index.astype(float).values
    aris = 1. / cdfs
    bf_quantiles = se.values.astype(float)
    marginal.params = theta_bestfit
    fs_quantiles = marginal.ppf(cdfs)
    iok = (bf_quantiles > censor) & (fs_quantiles > censor)
    quant_err = np.abs(np.log(bf_quantiles[iok])\
        -np.log(fs_quantiles[iok]))

    assert np.all(quant_err < 1e-8)


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         data_reader.STATIONS_BESTFIT)
@pytest.mark.parametrize("censoring", [False, True])
def test_mle_vs_bestfit(distname, stationid, censoring, allclose):
    if distname == "LogNormal":
        pytest.skip("Problem with LogNormal in bestfit")

    streamflow = get_ams(stationid)

    bestfit, censor = data_reader.read_bestfit_mle(stationid, censoring)
    bestfit = bestfit.loc[:, distname]

    marginal = marginals.factory(distname)

    # Set very wide prior scale to get max likelihood
    marginals.SHAPE1_PRIOR_SCALE_MAX = 1e100
    if marginal.has_shape:
        marginal.shape1_prior_scale = 1e100

    ll_mle, theta_mle, dcens, ncens, cov = marginal.maximum_posterior_estimate(streamflow,
                                                                               censor,
                                                                               nexplore=50000)
    theta_bestfit = np.array([float(bestfit["Location"]),
                              math.log(float(bestfit["Scale"])),
                              float(bestfit["Shape"])])

    # Take into account bestfit parameterisation
    if distname in ["LogPearson3"]:
        theta_bestfit[0] = math.log(10**theta_bestfit[0])
        theta_bestfit[1] = math.log(math.log(10**float(bestfit["Scale"])))

    elif distname == "LogNormal":
        mean, std = bestfit.iloc[:2].astype(float)
        mu = math.log(mean / math.sqrt(1 + std**2 / mean**2))
        sig = math.log(1 + std**2 / mean**2)
        theta_bestfit[0] = mu
        theta_bestfit[1] = math.log(sig)

    elif distname == "Gamma":
        th, sc = theta_bestfit[:2]
        sc = math.exp(sc)
        theta_bestfit[0] = th*sc
        theta_bestfit[1] = math.log(th)

    # Fix 2 param distributions
    if re.search("^(LogN|Norm|Gum|Gam)", distname):
        theta_bestfit[-1] = 0

    # Test if we get better MLE than bestfit
    ll_bestfit = -marginal.neglogpost(theta_bestfit, dcens, censor, ncens)
    assert ll_bestfit < ll_mle + 2e-2

    # Test if bestfit MLE is not too far behind
    assert abs(ll_bestfit - ll_mle) < 0.5

