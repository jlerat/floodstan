import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod
import math

import numpy as np
import pandas as pd

import pytest
import warnings

import importlib

from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import t as tstud

from floodstan import marginals, sample, report
from floodstan import univariate_censored_sampling

from test_sample_univariate import get_stationids, get_ams

SEED = 5446

FTESTS = Path(__file__).resolve().parent

def test_process_stan_diagnostic():
    fd = FTESTS / "stan_diag.txt"
    with fd.open("r") as fo:
        diag = fo.read()

    dd = report.process_stan_diagnostic(diag)
    kk = ["message", "treedepth", "divergence", "ebfmi", "effsamplesz", "rhat",
          "divergence_proportion"]
    for k in kk:
        assert k in dd

    assert dd["divergence_proportion"] == 0


def test_report(allclose):
    stationid = "201001"
    marginal = marginals.factory("GEV")
    y = get_ams(stationid)
    N = len(y)
    stan_nchains = 5

    # Configure stan data and initialisation
    sv = sample.StanSamplingVariable(marginal, y,
                                     ninits=stan_nchains)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    # Clean output folder
    LOGGER = sample.get_logger(stan_logger=False)
    fout = FTESTS / "report" / stationid
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample
    smp = univariate_censored_sampling(data=stan_data,
                                       chains=stan_nchains,
                                       seed=SEED,
                                       iter_warmup=5000,
                                       iter_sampling=500,
                                       output_dir=fout,
                                       inits=stan_inits)
    # Get sample data
    params = smp.draws_pd()

    # Run report without obs
    rep, _ = report.ams_report(sv.marginal, params) #, design_aris=[100])
    assert rep.shape == (12, 14)

    # Run report with obs
    years = np.arange(1973, 2022)
    obs = {year: y[year] for year in years}
    rep, _ = report.ams_report(sv.marginal, params, obs)
    assert rep.shape == (12 + 2 * len(obs), 14)


@pytest.mark.parametrize("nobs", [10, 100])
@pytest.mark.parametrize("coeffvar", [0.1, 5., 10.])
def test_normal_predictive_posterior(nobs, coeffvar, allclose):
    # Inference of normal params with normal-gamma conjugate prior
    # See Murphy, K. P. (n.d.). Conjugate Bayesian analysis of the Gaussian distribution.
    # We assume
    # lam = 1 / sig^2
    # lam ~ Gamma(a0, b0)
    # mu ~ Normal(mu0, (k0.lam)^-1)

    # .. prior specifications
    mu0 = 10. # Assume prior mean equal to 10
    k0 = 5 # Assume 5 "prior" obs

    ks0 = k0 # Assume 5 "prior" obs for lam (same than mu)
    lam0 = 1 / 5**2 # Assume prior std equal to 5
    a0 = ks0 / 2
    b0 = ks0 / 2 / lam0
    # i.e. lam ~ gamma.rvs(a=a0, scale=1./b0, size=1000)

    # .. assumed observed statistics
    mub = 15.
    sigb = mub * coeffvar
    S = nobs * sigb**2 # S = sum (xi-mub)^2

    # .. posterior
    mun = (k0 * mu0 + nobs * mub) / (k0 + nobs)
    kn = k0 + nobs
    an = a0 + nobs / 2
    bn = b0 + S / 2 + k0 * nobs * (mub - mu0)**2 / (k0 + nobs) / 2

    # .. posterior samples
    nsamples = 1000
    lamp = gamma.rvs(a=an, scale=1./bn, size=nsamples)
    mup = norm.rvs(loc=mun, scale=1/np.sqrt(lamp * kn), size=nsamples)
    params = pd.DataFrame({"ylocn": mup, "ylogscale": -np.log(lamp) / 2,
                           "yshape1": 0.})

    # Get expected quantiles and predictive dist
    # .. from report
    aris = np.arange(2, 1001)
    marginal = marginals.factory("Normal")
    rep, _ = report.ams_report(marginal, params, design_aris=aris)

    ppred = rep.POSTERIOR_PREDICTIVE.filter(regex="DESIGN")
    ppred = ppred.values
    expq = rep.MEAN.filter(regex="DESIGN").values

    # .. theoretical
    _, cdf, _, _ = report._prepare_design_aris(aris, 0.)
    sc = np.sqrt(bn * (kn + 1) / an / kn)
    ppred_th = tstud.ppf(cdf, loc=mun, scale=sc, df=2*an)

    # Check we can approximate posterior pred dist up to 1000 ari
    ari_max = aris[~np.isnan(ppred)][-1]
    assert ari_max == 1000

    # Check posterior predictive
    emax = np.abs(np.log(ppred) - np.log(ppred_th)).max()
    print(f"emax = {emax:6.3f}")

    atol, rtol = 1e-3, 5e-2
    idx = np.abs(ppred - ppred_th) > atol + rtol * ppred
    assert allclose(ppred, ppred_th, atol=atol, rtol=rtol)

