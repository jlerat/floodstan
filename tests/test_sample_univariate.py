import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal
from scipy.stats import ttest_1samp

import pytest
import warnings

from hydrodiy.plot import putils

import importlib
from tqdm import tqdm

from floodstan import marginals
from floodstan import sample
from floodstan import copulas
from floodstan import report
from floodstan import load_stan_model
from floodstan import univariate_censored_sampling

from floodstan import stan_test_marginal

from utils import get_stationids, get_ams, get_info
from utils import FTESTS

SEED = 5446
np.random.seed(SEED)

LOGGER = sample.get_logger(stan_logger=False)


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid", get_stationids())
def test_stan_sampling_variable(stationid, marginal_name, allclose):
    y = get_ams(stationid)
    censor = np.nanpercentile(y, 30)
    marginal = marginals.factory(marginal_name)

    sv = sample.StanSamplingVariable(marginal, y, censor)

    assert allclose(sv.censor, censor)
    assert allclose(sv.data, y)
    assert sv.N == len(y)
    assert sv.marginal_code == marginals.MARGINAL_NAMES[marginal_name]
    nhigh, nlow = (y>=censor).sum(), (y<censor).sum()
    Ncases = [[nhigh, 0, 0], [nlow, 0, 0], [0, 0, 0]]
    assert allclose(sv.Ncases, Ncases)

    i11 = np.where(y>=censor)[0]+1
    assert allclose(sv.i11, i11)

    assert sv.sampled_parameters.shape[0] == sample.NPARAMS_INITS_MAX

    dd = sv.to_dict()

    # Test writing to json
    fd = FTESTS / "sample.json"
    if fd.exists():
        fd.unlink()
    with fd.open("w") as fo:
        json.dump(dd, fo, indent=4)

    fd.unlink()

    # Test content of sample
    keys = ["ymarginal", "y",
            "ycensor",
            "ylocn_prior", "ylogscale_prior",
            "yshape1_prior"]
    for key in keys:
        assert key in dd

    sa = sv.stan_sample_args
    if marginal_name in sample.STAN_SAMPLE_ARGS:
        assert sa == sample.STAN_SAMPLE_ARGS[marginal_name]

    # Initial values
    sv = sample.StanSamplingVariable(marginal, y)
    d = sv.data
    inits = sv.initial_parameters
    for init in inits:
        for pn in marginals.PARAMETERS:
            assert f"y{pn}" in init

    cdfs = sv.initial_cdfs
    assert len(cdfs) == len(inits)
    assert len(cdfs[0]) == len(sv.data)


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("censoring", [False, True])
@pytest.mark.parametrize("stationid", ["203010"])
def test_univariate_censored_sampling(stationid, marginal_name, censoring, allclose):
    if stationid == "hard" and marginal_name == "Gamma":
        pytest.skip("Skipping sampling test. Data does not fit Gamma at all.")

    if marginal_name == "LogPearson3":
        pytest.skip("Skipping LogPearson3 due to numerical problems")

    y = get_ams(stationid)
    censor = np.nanmin(y) - 1
    if censoring:
        pcens = 5 if stationid == "hard" else 20
        censor = np.nanpercentile(y, pcens)

    marginal = marginals.factory(marginal_name)

    # Set STAN
    stan_nwarm = 5000
    stan_nsamples = 5000
    stan_nchains = 5

    # Prepare sampling data
    sv = sample.StanSamplingVariable(marginal, y, censor,
                                     ninits=stan_nchains)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters
    stan_args = sv.stan_sample_args

    # Test initial parameters are legit
    m = sv.marginal
    inocens = y >= censor
    for p in stan_inits:
        m.params = {k[1:]: v for k, v  in p.items()}
        lp = m.logpdf(y[inocens])
        assert np.all(np.isfinite(lp))

        lc = m.logcdf(y[inocens])
        assert np.all(np.isfinite(lc))

        stan_data["ylocn"] = m.locn
        stan_data["ylogscale"] = m.logscale
        stan_data["yshape1"] = m.shape1
        smp = stan_test_marginal(data=stan_data)

        # .. testing stan is matching py like in test_stan_functions.py
        lps = smp.filter(regex="luncens").values[inocens]
        assert allclose(lp, lps, atol=1e-5)

        lcs = smp.filter(regex="^lcens").values[inocens]
        assert allclose(lc, lcs, atol=1e-5)

    # Clean output folder
    fname = f"univariate_{stationid}_{marginal_name}_{censoring}"
    fout = FTESTS / "sampling" / fname
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              parallel_chains=stan_nchains,
              iter_warmup=stan_nwarm)
    kw.update(stan_args)

    # Sample
    smp = univariate_censored_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    for f in fout.glob("*.*"):
        f.unlink()
    fout.rmdir()

    # Test diag
    assert diag["effsamplesz"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"

    # Test divergence
    prc = diag["divergence_proportion"]
    print(f"\n{stationid}-{marginal_name}-{censoring} : Divergence proportion = {prc}\n")
    if not marginal.name in ["LogPearson3", "GeneralizedPareto",
                             "GeneralizedLogistic"]:
        thresh = 50
        assert prc < thresh

    # Test report
    rep, _ = report.ams_report(marginal, df)
    rep = rep.filter(regex="DESIGN", axis=0)
    rep = rep.filter(regex="MEAN|PREDICTIVE", axis=1)
    assert rep.notnull().all().all()


def test_univariate_censored_sampling_not_enough_data(allclose):
    stationid = "201001"
    y = get_ams(stationid).values
    censor = np.nanmin(y) - 1
    marginal = marginals.factory("GEV")

    # Set STAN
    stan_nwarm = 10000
    stan_nsamples = 5000
    stan_nchains = 5

    # Prepare sampling data
    sv = sample.StanSamplingVariable(marginal, y, censor,
                                     ninits=stan_nchains)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    # Leaves only 4 values not nan
    inonan = np.where(~np.isnan(y))[0]
    stan_data["y"][inonan[4:]] = np.nan

    msg = "Error during sampling"
    with pytest.raises(RuntimeError, match=msg):
        smp = univariate_censored_sampling(data=stan_data,
                                           seed=SEED,
                                           iter_sampling=stan_nsamples // stan_nchains,
                                           inits=stan_inits,
                                           chains=stan_nchains,
                                           parallel_chains=stan_nchains,
                                           iter_warmup=stan_nwarm)


def test_univariate_censored_sampling_not_enough_inits(allclose):
    stationid = "201001"
    y = get_ams(stationid).values
    censor = np.nanmin(y) - 1
    marginal = marginals.factory("GEV")

    # Set STAN
    stan_nwarm = 10000
    stan_nsamples = 5000
    stan_nchains = 5

    # Prepare sampling data
    sv = sample.StanSamplingVariable(marginal, y, censor,
                                     ninits=stan_nchains)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters[:3]

    msg = "Expected 1 or"
    with pytest.raises(ValueError, match=msg):
        smp = univariate_censored_sampling(data=stan_data,
                                           seed=SEED,
                                           iter_sampling=stan_nsamples // stan_nchains,
                                           inits=stan_inits,
                                           chains=stan_nchains,
                                           parallel_chains=stan_nchains,
                                           iter_warmup=stan_nwarm)


def test_univariate_klemes_data(allclose):
    fd = FTESTS / "data" / "Klemes_19986_dilletantism_fig5_plot_data.csv"
    y = pd.read_csv(fd).iloc[:, 1].sort_values().values
    marginal_name = "GEV"
    marginal = marginals.factory(marginal_name)
    censor = 0.

    # Set STAN
    stan_nwarm = 10000
    stan_nsamples = 5000
    stan_nchains = 5

    # Prepare sampling data
    sv = sample.StanSamplingVariable(marginal, y, censor,
                                     ninits=stan_nchains)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters
    stan_args = sv.stan_sample_args

    # Clean output folder
    fname = f"univariate_klemes"
    fout = FTESTS / "sampling" / fname
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    design = {}
    for ibias in ["low", "neutral", "high"]:
        yb = y.copy()
        if ibias == "low":
            yb[:3] = 3.
        elif ibias == "high":
            yb[:3] = 7.5

        stan_data["y"] = yb

        # Sample arguments
        kw = dict(data=stan_data,
                  seed=SEED,
                  iter_sampling=stan_nsamples // stan_nchains,
                  output_dir=fout,
                  inits=stan_inits,
                  chains=stan_nchains,
                  parallel_chains=stan_nchains,
                  iter_warmup=stan_nwarm)
        kw.update(stan_args)

        # Sample
        smp = univariate_censored_sampling(**kw)
        df = smp.draws_pd()
        diag = report.process_stan_diagnostic(smp.diagnose())
        rep, _ = report.ams_report(marginal, df)

        for f in fout.glob("*.*"):
            f.unlink()

        # Test diag
        assert diag["effsamplesz"] == "satisfactory"
        assert diag["rhat"] == "satisfactory"

        design[ibias] = rep.POSTERIOR_PREDICTIVE.filter(regex="DESIGN")

    fout.rmdir()
    design = pd.DataFrame(design)
    diff = math.log(design.high.DESIGN_ARI100) - math.log(design.low.DESIGN_ARI100)
    assert diff < 5e-2
