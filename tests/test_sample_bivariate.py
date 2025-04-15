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

from floodstan import marginals
from floodstan import stan_test_marginal, stan_test_copula

from floodstan import report, sample
from floodstan import bivariate_censored_sampling

from test_sample_univariate import get_stationids, get_ams

SEED = 5446
np.random.seed(SEED)

FTESTS = Path(__file__).resolve().parent

STATIONIDS = get_stationids()

LOGGER = sample.get_logger(stan_logger=False)


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
def test_stan_sampling_dataset(distname, allclose):
    y = get_ams("203010")
    z = get_ams("201001")
    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    y, z = df.y, df.z

    marginal = marginals.factory(distname)

    yv = sample.StanSamplingVariable(marginal, y, 100)
    zv = sample.StanSamplingVariable(marginal, z, 100)
    dset = sample.StanSamplingDataset([yv, zv], "Gaussian")

    assert dset.copula_name == "Gaussian"
    assert allclose(dset.Ncases, [[38, 3, 1], [6, 7, 0], [9, 1, 0]])

    dd = dset.to_dict()
    keys = ["marginal", "censor", "locn_prior",
            "logscale_prior", "shape1_prior"]
    for name, key in prod(["y", "z"], keys):
        assert f"{name}{key}" in dd

    i11 = dset.i11
    assert pd.notnull(df.y.iloc[i11-1]).all()
    assert pd.notnull(df.z.iloc[i11-1]).all()

    i31 = dset.i31
    assert pd.isnull(df.y.iloc[i31-1]).all()
    assert pd.notnull(df.z.iloc[i31-1]).all()

    # Initial values
    inits = dset.initial_parameters
    for init in inits:
        assert "rho" in init
        for pn in ["locn", "logscale", "shape1"]:
            for n in ["y", "z"]:
                assert f"{n}{pn}" in init


@pytest.mark.parametrize("copula", sample.COPULA_NAMES_STAN)
@pytest.mark.parametrize("censoring", [False, True])
def test_bivariate_sampling_satisfactory(copula, censoring, allclose):
    LOGGER = sample.get_logger(stan_logger=True)

    marginal = marginals.factory("GEV")

    stan_nwarm = 10000
    stan_nsamples = 5000
    stan_nchains = 5

    stationid = STATIONIDS[0]
    y = get_ams(stationid)

    # Generate random covariate
    scale = np.nanstd(y) / 5
    z = y + np.random.normal(0, scale, size=len(y))

    N = len(y)

    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    y, z = df.y, df.z

    censor = y.median() if censoring else np.nanmin(y) - 1.
    yv = sample.StanSamplingVariable(marginal, y, censor,
                                     ninits=stan_nchains)

    censor = z.median() if censoring else np.nanmin(z) - 1.
    zv = sample.StanSamplingVariable(marginal, z, censor,
                                     ninits=stan_nchains)

    sv = sample.StanSamplingDataset([yv, zv], copula)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    fout_stan = FTESTS / "sampling" / "bivariate" / f"{stationid}_{copula}"
    fout_stan.mkdir(exist_ok=True, parents=True)
    for f in fout_stan.glob("*.*"):
        f.unlink()

    smp = bivariate_censored_sampling(data=stan_data,
                                      chains=stan_nchains,
                                      seed=SEED,
                                      iter_warmup=stan_nwarm,
                                      iter_sampling=
                                      stan_nsamples//stan_nchains,
                                      output_dir=fout_stan,
                                      inits=stan_inits,
                                      show_progress=False)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    # Test diag
    assert diag["treedepth"] == "satisfactory"
    assert diag["ebfmi"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"

    # Test divergence
    prc = diag["divergence_proportion"]
    thresh = 30 if censoring else 5
    assert prc < thresh

    # Clean folder
    for f in fout_stan.glob("*.*"):
        f.unlink()

    fout_stan.rmdir()


def test_bivariate_sampling_problem(allclose):
    LOGGER = sample.get_logger(stan_logger=True)

    stan_nwarm = 10000
    stan_nsamples = 5000
    stan_nchains = 10

    LOGGER = sample.get_logger(stan_logger=True)

    fd = FTESTS / "data" / "bivariate_stan_data.json"
    with fd.open("r") as fo:
        stan_data = json.load(fo)

    fi = FTESTS / "data" / "bivariate_stan_inits.json"
    with fd.open("r") as fo:
        stan_inits = json.load(fo)

    fout_stan = FTESTS / "sampling" / "bivariate" / "problem"
    fout_stan.mkdir(exist_ok=True, parents=True)
    for f in fout_stan.glob("*.*"):
        f.unlink()

    smp = bivariate_censored_sampling(data=stan_data,
                                      chains=stan_nchains,
                                      seed=SEED,
                                      iter_warmup=stan_nwarm,
                                      iter_sampling=
                                      stan_nsamples//stan_nchains,
                                      output_dir=fout_stan,
                                      inits=stan_inits,
                                      show_progress=False)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    # Test diag
    assert diag["treedepth"] == "satisfactory"
    assert diag["ebfmi"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"

    # Test divergence
    prc = diag["divergence_proportion"]
    assert prc < 20

    # Clean folder
    for f in fout_stan.glob("*.*"):
        f.unlink()

    fout_stan.rmdir()


@pytest.mark.parametrize("varname", ["y", "z"])
def test_bivariate_sampling_not_enough_data(varname, allclose):
    LOGGER = sample.get_logger(stan_logger=True)

    marginal = marginals.factory("GEV")

    stan_nwarm = 10000
    stan_nsamples = 5000
    stan_nchains = 5

    stationid = STATIONIDS[0]
    y = get_ams(stationid)

    # Generate random covariate
    scale = np.nanstd(y) / 5
    z = y + np.random.normal(0, scale, size=len(y))

    N = len(y)

    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    y, z = df.y, df.z

    censor = y.median()
    yv = sample.StanSamplingVariable(marginal, y, censor,
                                     ninits=stan_nchains)

    censor = z.median()
    zv = sample.StanSamplingVariable(marginal, z, censor,
                                     ninits=stan_nchains)

    sv = sample.StanSamplingDataset([yv, zv], "Gumbel")
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    # Leaves only 4 values not nan
    if varname == "y":
        inonan = np.where(~np.isnan(y))[0]
        stan_data["y"][inonan[4:]] = np.nan
    else:
        inonan = np.where(~np.isnan(z))[0]
        stan_data["z"][inonan[4:]] = np.nan


    msg = "Error during sampling"
    with pytest.raises(RuntimeError, match=msg):
        smp = bivariate_censored_sampling(data=stan_data,
                                      chains=stan_nchains,
                                      seed=SEED,
                                      iter_warmup=stan_nwarm,
                                      iter_sampling=
                                      stan_nsamples//stan_nchains,
                                      inits=stan_inits,
                                      show_progress=False)

