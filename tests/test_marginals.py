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


@pytest.mark.parametrize("param_name",
                         marginals.PARAMETERS)
def test_prior_properties(param_name):
    with pytest.raises(ValueError, match="Expected param_name"):
        marginals.TruncatedNormalParameterPrior("bidule")

    prior = marginals.TruncatedNormalParameterPrior(param_name)
    str(prior)
    assert len(prior.to_list()) == 2

    s = prior.sample(100000)
    assert np.all(s > prior.lower)
    assert np.all(s < prior.upper)

    x0 = getattr(marginals, f"{param_name.upper()}_LOWER") + 1e-3
    x1 = getattr(marginals, f"{param_name.upper()}_UPPER") - 1e-3
    x = np.linspace(x0, x1, 10)
    assert np.all(np.isfinite(prior.logpdf(x)))
    assert np.all(np.isfinite(prior.logcdf(x)))

    prior.lower = 10.
    prior.upper = -10.
    with pytest.raises(ValueError, match="Inconsistent lower"):
        prior.lower

    prior.set_uninformative()
    rv = prior.rv
    assert prior.loc == 0
    assert prior.scale == 1e10
    assert prior._a == -1.
    assert prior._b == 1.

    pc = prior.clone()
    for n in ["lower", "upper", "loc", "scale", "uninformative"]:
        v1 = getattr(pc, n)
        v2 = getattr(prior, n)
        assert v1 == v2


def test_floodfreqdist(allclose):
    name = "bidule"
    dist = marginals.FloodFreqDistribution(name)

    assert dist.name == name
    assert hasattr(dist, "locn")
    assert hasattr(dist, "logscale")
    assert hasattr(dist, "shape1")
    assert hasattr(dist, "locn_prior")
    assert hasattr(dist, "logscale_prior")
    assert hasattr(dist, "shape1_prior")

    s = str(dist)
    assert isinstance(s, str)

    dist.locn = 10
    assert dist.locn == 10

    for pn in marginals.PARAMETERS:
        dist[pn] = 1
        assert getattr(dist, pn) == 1

        dist[pn] = -1
        assert getattr(dist, pn) == -1

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
        dist.params = [10, 1, np.nan]

    with pytest.raises(AttributeError, match="can't set"):
        dist.locn_prior = "bidule"

    with pytest.raises(ValueError, match="Invalid"):
        dist.locn_prior.scale = np.nan


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
def test_floodfreqdist_clone(marginal_name, allclose):
    marginal = marginals.factory(marginal_name)
    streamflow = get_ams("203014")
    marginal.params_guess(streamflow)

    for n in marginals.PARAMETERS:
        prior = getattr(marginal, f"{n}_prior")
        prior.loc = marginal[n]

    mc = marginal.clone()
    for n in marginals.PARAMETERS:
        assert marginal[n] == mc[n]

        prior = getattr(mc, f"{n}_prior")
        assert prior.loc == mc[n]


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
def test_marginals_properties(marginal_name, allclose):
    streamflow = get_ams("203014")
    dist = marginals.factory(marginal_name)
    dist.params_guess(streamflow)
    s = str(dist)
    y = dist.rvs(size=10000)
    p = dist.cdf(y)
    expected = dist.ppf(p)
    assert allclose(expected, y)



@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         get_stationids())
def test_marginals_vs_nrivfloodfreq(marginal_name, stationid, allclose):
    nparams = 500
    streamflow = get_ams(stationid)

    if marginal_name in ["GeneralizedPareto", "GeneralizedLogistic", "Gamma"]:
        pytest.skip(f"Skipping {marginal_name}")

    dist1 = fdist.factory(marginal_name)
    dist2 = marginals.factory(marginal_name)

    # Test lh moments
    p = dist1.fit_lh_moments(streamflow).iloc[0]
    dist1.set_dict_params(p.to_dict())

    # Skip if extreme shape parameter values
    if marginal_name in ["GEV", "LogPearson3"]:
        sh = dist1.kappa if marginal_name == "GEV" else dist1.g
        if sh < marginals.SHAPE1_LOWER or \
                sh > marginals.SHAPE1_UPPER:
            pytest.skip("Skipping test because shape is out of bounds")

    dist2.fit_lh_moments(streamflow)

    if marginal_name == "LogNormal":
        assert allclose(dist2.locn, dist1.m, atol=1e-6)
        assert allclose(dist2.logscale, math.log(dist1.s), atol=1e-6)

    elif marginal_name == "Normal":
        assert allclose(dist2.locn, dist1.mu, atol=1e-6)
        assert allclose(dist2.logscale, dist1.logsig, atol=1e-6)

    elif marginal_name == "Gumbel":
        assert allclose(dist2.locn, dist1.tau, atol=1e-6)
        assert allclose(dist2.logscale, dist1.logalpha, atol=1e-6)

    elif marginal_name == "GEV":
        pass
        # Upgraded GEV lh moments
        #assert allclose(dist2.locn, dist1.tau, atol=1e-6)
        #assert allclose(dist2.logscale, dist1.logalpha, atol=1e-6)
        #assert allclose(dist2.shape1, dist1.kappa, atol=1e-6)

    elif marginal_name == "LogPearson3":
        assert allclose(dist2.locn, dist1.m, atol=1e-6)
        assert allclose(dist2.logscale, math.log(dist1.s), atol=1e-6)
        assert allclose(dist2.shape1, dist1.g, atol=1e-6)

    params, _ = fsample.bootstrap_lh_moments(dist1, streamflow, nparams)
    for _, param in params.iterrows():
        # Set parameters
        dist1.set_dict_params(param.to_dict())

        # Skip if extreme shape parameter values
        if marginal_name in ["GEV", "LogPearson3"]:
            sh = dist1.kappa if marginal_name == "GEV" else dist1.g
            if sh < marginals.SHAPE1_LOWER or \
                    sh > marginals.SHAPE1_UPPER:
                continue

        if marginal_name == "LogNormal":
            dist2.locn = dist1.m
            dist2.logscale = math.log(dist1.s)

        elif marginal_name == "Normal":
            dist2.locn = dist1.mu
            dist2.logscale = dist1.logsig

        elif marginal_name == "Gumbel":
            dist2.locn = dist1.tau
            dist2.logscale = dist1.logalpha

        elif marginal_name == "GEV":
            dist2.locn = dist1.tau
            dist2.logscale = dist1.logalpha
            dist2.shape1 = dist1.kappa

        elif marginal_name == "LogPearson3":
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
        atol = 5e-7
        assert allclose(cdf1, cdf2, atol=atol)

        lcdf1 = dist1.logcdf(streamflow)
        lcdf2 = dist2.logcdf(streamflow)
        assert allclose(lcdf1, lcdf2, atol=atol)


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         get_stationids())
def test_params_guess(marginal_name, stationid, allclose):
    nvalues = 1000
    nboot = 200
    streamflow = get_ams(stationid)

    dist = marginals.factory(marginal_name)
    dist.params_guess(streamflow)

    distb = marginals.factory(marginal_name)
    for iboot in range(nboot):
        ys = dist.rvs(nvalues)
        distb.params_guess(ys)

        ymin, ymax = distb.support
        assert ymin <= ys.min()
        assert ymax >= ys.max()

        lpdf = distb.logpdf(ys)
        assert np.all(~np.isnan(lpdf))


@pytest.mark.parametrize("marginal_name",
                         data_reader.DISTRIBUTIONS)
@pytest.mark.parametrize("station",
                         data_reader.STATIONS)
def test_quantile_vs_flike(marginal_name, station, allclose):
    try:
        testdata, fr = data_reader.get_test_data(station, marginal_name,
                                                 "LH0", False, "flike")
    except FileNotFoundError:
        pytest.skip("No test data.")

    qt = testdata["quantiles"].loc[:, "quantile"]
    pp = 1-1/qt.index

    dist = marginals.factory(marginal_name)

    # Set parameters
    fit = testdata["fit"]

    if marginal_name == "GEV":
        dist.locn = fit.loc[0, 1]
        dist.logscale = math.log(fit.loc[1, 1])
        dist.shape1 = fit.loc[2, 1]

    elif marginal_name == "LogPearson3":
        dist.locn = fit.loc[0, 1]
        dist.logscale = math.log(fit.loc[1, 1])
        dist.shape1 = fit.loc[2, 1]

    elif marginal_name == "Gumbel":
        dist.locn = fit.loc[0, 1]
        dist.logscale = math.log(fit.loc[1, 1])

    elif marginal_name == "LogNormal":
        dist.locn = fit.loc[0, 1]
        dist.logscale = math.log(fit.loc[1, 1])

    qt2 = dist.ppf(pp)
    if marginal_name in ["LogNormal", "LogPearson3"]:
        # flike operates on log10 transform data for these 2 distributions
        # and not log transform. As our procedure apply a log transform,
        # we first transform to log10 and then exponentiate, which
        # leads to log10 transform data within our code.
        qt2 = 10**np.log(qt2)

    assert allclose(qt, qt2, atol=0.1, rtol=5e-3)


@pytest.mark.parametrize("marginal_name",
                         data_reader.DISTRIBUTIONS)
@pytest.mark.parametrize("station",
                         data_reader.STATIONS)
@pytest.mark.parametrize("censoring", [True, False])
def test_fit_lh_moments_vs_flike(marginal_name, station, censoring, allclose):
    eta = 0
    try:
        testdata, fr = data_reader.get_test_data(station, marginal_name, \
                                f"LH{eta}", censoring, "flike")
    except FileNotFoundError:
        pytest.skip("No test data.")

    streamflow = testdata["data"].streamflow
    fit = testdata["fit"]

    dist = marginals.factory(marginal_name)
    if marginal_name in ["LogNormal", "LogPearson3"]:
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
    if marginal_name == "GEV":
        assert allclose(samples.locn, fit.loc[0, 1], rtol=5e-3, atol=1e-2)
        assert allclose(samples.logscale, math.log(fit.loc[1, 1]), rtol=0, atol=1e-2)
        assert allclose(samples.shape1, fit.loc[2, 1], rtol=0, atol=1e-2)

    elif marginal_name == "LogPearson3":
        assert allclose(samples.locn, fit.loc[0, 1], rtol=0, atol=1e-2)
        assert allclose(samples.scale, fit.loc[1, 1], rtol=0, atol=1e-2)

        # 2 datasets showing slightly higher error than others
        # probably due to rounding values in flike.
        if fr.stem in ["203014_LogPearson3_LH0_censoring", \
                        "arr84_LogPearson3_LH0_censoring"]:
            assert allclose(samples.shape1, fit.loc[2, 1], rtol=0, atol=2e-2)
        else:
            assert allclose(samples.shape1, fit.loc[2, 1], rtol=0, atol=1e-2)

    elif marginal_name == "Gumbel":
        assert allclose(samples.logscale, math.log(fit.loc[1, 1]), rtol=0,atol=1e-2)
        assert allclose(samples.locn, fit.loc[0, 1], rtol=0, atol=1e-1)

    elif marginal_name == "LogNormal":
        assert allclose(samples.locn, fit.loc[0, 1], rtol=0, atol=1e-2)
        assert allclose(samples.scale, fit.loc[1, 1], rtol=0, atol=1e-2)


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid", get_stationids()[:3])
@pytest.mark.parametrize("censoring", [False, True])
def test_marginals_maxpost_numerical(marginal_name, stationid, censoring, allclose):
    streamflow = get_ams(stationid)
    marginal = marginals.factory(marginal_name)
    censor = streamflow.quantile(0.33) if censoring else -1e10
    lmp, theta_lmp, dcens, ncens = \
        marginal.maximum_posterior_estimate(streamflow, censor)

    cov = np.diag(theta_lmp ** 2)

    # Sample params from laplace approx
    params = np.random.multivariate_normal(mean=theta_lmp,
                                           cov=cov, size=5000)
    for theta in params:
        lp = -marginal.neglogpost(theta, dcens, censor, ncens)
        if np.isfinite(lp) and not np.isnan(lp):
            mess = f"[{marginal_name}/{stationid}/{censoring}] "\
                   + f" num: lp={lp:0.1e} theta={theta} "\
                   + f" lmp: lp={lmp:0.1e} theta={theta_lmp}"
            eps = 5e-2
            assert lp < lmp + eps, print(mess)


@pytest.mark.parametrize("marginal_name", ["Gamma", "LogNormal", "Normal"])
@pytest.mark.parametrize("stationid", get_stationids())
def test_marginals_mle_theoretical(marginal_name, stationid, allclose):
    streamflow = get_ams(stationid)
    marginal = marginals.factory(marginal_name)

    # Uninformative prior to get mle
    marginal.locn_prior.set_uninformative()
    marginal.logscale_prior.set_uninformative()
    marginal.shape1_prior.set_uninformative()

    mlp_mle, theta_mle, dcens, ncens = \
        marginal.maximum_posterior_estimate(streamflow)

    # Theoretical value of MLE
    if marginal_name == "LogNormal":
        marginal.params_guess(streamflow)
        locn, logscale = marginal.params[:2]

        # Check lognorm mle
        locn0 = np.log(streamflow).mean()
        logscale0 = math.log(math.sqrt(((np.log(streamflow)-locn0)**2).mean()))
        assert allclose(locn, locn0, atol=1e-8, rtol=0.)
        assert allclose(logscale, logscale0, atol=1e-8, rtol=0.)

    elif marginal_name == "Normal":
        marginal.params_guess(streamflow)
        locn, logscale = marginal.params[:2]

        # Check norm mle
        locn0 = streamflow.mean()
        logscale0 = math.log(math.sqrt(((streamflow-locn0)**2).mean()))
        assert allclose(locn, locn0, atol=1e-8, rtol=0.)
        assert allclose(logscale, logscale0, atol=1e-8, rtol=0.)

    elif marginal_name == "Gamma":
        marginal.params_guess(streamflow)
        locn, logscale = marginal.params[:2]

    assert allclose(theta_mle[0], locn, atol=1e-5, rtol=1e-5)
    assert allclose(theta_mle[1], logscale, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         data_reader.STATIONS_BESTFIT)
@pytest.mark.parametrize("censoring", [False, True])
def test_marginals_vs_bestfit(marginal_name, stationid, censoring, allclose):
    if marginal_name == "LogNormal":
        pytest.skip("Problem with LogNormal in bestfit")
    streamflow = get_ams(stationid)

    bestfit, censor = data_reader.read_bestfit_mle(stationid, censoring)
    bestfit = bestfit.loc[:, marginal_name]

    marginal = marginals.factory(marginal_name)

    theta_bestfit = np.array([float(bestfit["Location"]),
                              math.log(float(bestfit["Scale"])),
                              float(bestfit["Shape"])])

    # Take into account bestfit parameterisation
    if marginal_name in ["LogPearson3"]:
        theta_bestfit[0] = math.log(10**theta_bestfit[0])
        theta_bestfit[1] = math.log(math.log(10**float(bestfit["Scale"])))

    elif marginal_name == "LogNormal":
        mean, std = bestfit.iloc[:2].astype(float)
        mu = math.log(mean / math.sqrt(1 + std**2 / mean**2))
        sig = math.log(1 + std**2 / mean**2)
        theta_bestfit[0] = mu
        theta_bestfit[1] = math.log(sig)

    elif marginal_name == "Gamma":
        th, sc = theta_bestfit[:2]
        sc = math.exp(sc)
        theta_bestfit[0] = th*sc
        theta_bestfit[1] = math.log(th)

    # Fix 2 param distributions
    if re.search("^(LogN|Norm|Gum|Gam)", marginal_name):
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


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         data_reader.STATIONS_BESTFIT)
@pytest.mark.parametrize("censoring", [False, True])
def test_mle_vs_bestfit(marginal_name, stationid, censoring, allclose):
    if marginal_name == "LogNormal":
        pytest.skip("Problem with LogNormal in bestfit")

    streamflow = get_ams(stationid)

    bestfit, censor = data_reader.read_bestfit_mle(stationid, censoring)
    bestfit = bestfit.loc[:, marginal_name]

    marginal = marginals.factory(marginal_name)

    # Uninformative prior to get mle
    marginal.locn_prior.set_uninformative()
    marginal.logscale_prior.set_uninformative()
    marginal.shape1_prior.set_uninformative()

    ll_mle, theta_mle, dcens, ncens = \
        marginal.maximum_posterior_estimate(streamflow, censor)

    # Get bestfit param
    # + take into account bestfit parameterisation
    theta_bestfit = np.array([float(bestfit["Location"]),
                              math.log(float(bestfit["Scale"])),
                              float(bestfit["Shape"])])

    if marginal_name in ["LogPearson3"]:
        theta_bestfit[0] = math.log(10**theta_bestfit[0])
        theta_bestfit[1] = math.log(math.log(10**float(bestfit["Scale"])))

    elif marginal_name == "LogNormal":
        mean, std = bestfit.iloc[:2].astype(float)
        mu = math.log(mean / math.sqrt(1 + std**2 / mean**2))
        sig = math.log(1 + std**2 / mean**2)
        theta_bestfit[0] = mu
        theta_bestfit[1] = math.log(sig)

    elif marginal_name == "Gamma":
        th, sc = theta_bestfit[:2]
        sc = math.exp(sc)
        theta_bestfit[0] = th*sc
        theta_bestfit[1] = math.log(th)

    # Fix 2 param distributions
    if re.search("^(LogN|Norm|Gum|Gam)", marginal_name):
        theta_bestfit[-1] = 0

    # Test if we get better MLE than bestfit
    ll_bestfit = -marginal.neglogpost(theta_bestfit, dcens, censor, ncens)
    assert ll_bestfit < ll_mle + 1.

    # Test if bestfit MLE is not too far behind
    assert abs(ll_bestfit - ll_mle) < 1.0


def test_gev_shape0(allclose):
    gev = marginals.GEV()
    gev.params = [100, math.log(50), 0.]

    gum = marginals.Gumbel()
    gum.params = gev.params

    s0, s1 = gev.support
    assert np.isinf(s0)
    assert np.isinf(s1)

    eps = 1e-8
    u = np.linspace(eps, 1-eps, 1000)
    qa = gev.ppf(u)
    qb = gum.ppf(u)
    assert allclose(qa, qb)

    q0 = qa.min()
    q1 = qa.max()
    qq = np.linspace(q0, q1, 1000)
    ca = gev.cdf(qq)
    cb = gum.cdf(qq)
    allclose(ca, cb)


def test_lp3_shape0(allclose):
    lp3 = marginals.LogPearson3()
    lp3.params = [100, math.log(50), 0.]

    ln = marginals.LogNormal()
    ln.params = lp3.params

    s0, s1 = lp3.support
    assert np.isinf(s0)
    assert np.isinf(s1)

    eps = 1e-8
    u = np.linspace(eps, 1-eps, 1000)
    qa = lp3.ppf(u)
    qb = ln.ppf(u)
    assert allclose(qa, qb)

    q0 = qa.min()
    q1 = qa.max()
    qq = np.linspace(q0, q1, 1000)
    ca = lp3.cdf(qq)
    cb = ln.cdf(qq)
    allclose(ca, cb)

