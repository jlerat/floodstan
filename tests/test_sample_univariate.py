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

from floodstan import marginals, sample, copulas
from floodstan import univariate_censored_sampling

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent

# --- Utils functions ----------------------------
def get_stationids(skip=5):
    fs = FTESTS / "data"
    stationids = []
    for ifile, f in enumerate(fs.glob("*_AMS.csv")):
        if not ifile % skip == 0:
            continue

        sid = re.sub("_.*", "", f.stem)
        if re.search("LIS", sid):
            continue
        stationids.append(sid)

    return stationids

def get_ams(stationid):
    fs = FTESTS / "data" / f"{stationid}_AMS.csv"
    df = pd.read_csv(fs, skiprows=15, index_col=0)
    return df.iloc[:, 0]

def get_info():
    fs = FTESTS / "data" / "stations.csv"
    df = pd.read_csv(fs, skiprows=17)

    df.columns = [re.sub(" |,", "_", re.sub(" \\(.*", "", cn)) \
                            for cn in df.columns]
    df.loc[:, "Station_ID"] = df.Station_ID.astype(str)
    df = df.set_index("Station_ID")
    stationids = get_stationids()
    df = df.loc[stationids, :]

    return df


# ------------------------------------------------

def test_stan_sampling_variable(allclose):
    y = get_ams("203010")
    sv = sample.StanSamplingVariable()
    censor = y.median()

    msg = "Data has not been set."
    with pytest.raises(ValueError, match=msg):
        d = sv.data

    msg = "Cannot find"
    with pytest.raises(ValueError, match=msg):
        sv.set_marginal("bidule")

    msg = "Expected data"
    with pytest.raises(ValueError, match=msg):
        sv.set_data(y.values[:, None], censor)

    sv = sample.StanSamplingVariable(y, "GEV", censor)
    assert allclose(sv.censor, censor)
    assert isinstance(sv.marginal, marginals.FloodFreqDistribution)
    assert allclose(sv.data, y)
    assert sv.N == len(y)
    assert sv.marginal_code == 3
    assert sv.marginal_name == "GEV"
    nhigh, nlow = (y>=censor).sum(), (y<censor).sum()
    assert allclose(sv.Ncases, [[nhigh, 0, 0], [nlow, 0, 0], [0, 0, 0]])

    i11 = np.where(y>=censor)[0]+1
    assert allclose(sv.i11, i11)

    dd = sv.to_dict()
    keys = ["ymarginal", "y",
            "ycensor",
            "ylocn_prior", "ylogscale_prior",
            "yshape1_prior"]
    for key in keys:
        assert key in dd

    assert allclose(dd["yshape1_prior"], sample.SHAPE1_PRIOR)

    # Rapid setting
    sv = sample.StanSamplingVariable(y)
    msg = "Initial parameters"
    with pytest.raises(ValueError, match=msg):
        ip = sv.initial_parameters

    sv = sample.StanSamplingVariable(y, "GEV")
    d = sv.data

    # Initial values
    inits = sv.initial_parameters
    for pn in ["locn", "logscale", "shape1"]:
        assert pn in inits


def test_stan_sampling_dataset(allclose):
    y = get_ams("203010")
    z = get_ams("201001")
    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    y, z = df.y, df.z

    yv = sample.StanSamplingVariable(y, "GEV", 100)
    zv = sample.StanSamplingVariable(z, "GEV", 100)
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
    assert "rho" in inits
    for pn in ["locn", "logscale", "shape1"]:
        for n in ["y", "z"]:
            assert f"{n}{pn}" in inits


def test_univariate_sampling_short_syntax(allclose):
    stationids = get_stationids()
    stationid = stationids[0]
    marginal = "GEV"
    y = get_ams(stationid)

    # Set STAN
    sv = sample.StanSamplingVariable(y, marginal)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    # Clean output folder
    fout = FTESTS / "sampling" / "univariate_short_syntax"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample
    smp = univariate_censored_sampling(data=stan_data,
                                       inits=stan_inits,
                                       output_dir=fout)
    df = smp.draws_pd()


@pytest.mark.parametrize("marginal",
                         list(sample.MARGINAL_NAMES.keys()))
@pytest.mark.parametrize("stationid",
                         get_stationids()[:3])
def test_univariate_sampling(marginal, stationid, allclose):
    # Testing univariate sampling following the process described by
    # Samantha R Cook, Andrew Gelman & Donald B Rubin (2006)
    # Validation of Software for Bayesian Models Using Posterior Quantiles,
    # Journal of Computational and Graphical Statistics, 15:3, 675-692,
    # DOI: 10.1198/106186006X136976

    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 100
    nrepeat = 30
    nrows, ncols = 3, 3
    axwidth, axheight = 5, 5
    print("\n")

    y = get_ams(stationid)
    N = len(y)

    # Setup image
    plt.close("all")
    w, h = axwidth*ncols, axheight*nrows
    fig, ax = plt.subplots(figsize=(w, h), layout="tight")

    if marginal in ["Gumbel", "LogNormal", "Normal", "Gamma"]:
        parnames = ["locn", "logscale"]
    else:
        parnames = ["locn", "logscale", "shape1"]

    dist = marginals.factory(marginal)
    dist.params_guess(y)

    if dist.shape1<marginals.SHAPE1_LOWER or dist.shape1>marginals.SHAPE1_UPPER:
        pytest.skip("Shape parameter outside of accepted bounds.")

    # Prior distribution centered around dist params
    ylocn_prior = [dist.locn, abs(dist.locn)*0.5]
    ylogscale_prior = [dist.logscale, abs(dist.logscale)*0.5]
    yshape1_prior = [max(0.1, dist.shape1), 0.2]

    test_stat = []
    # .. double the number of tries to get nrepeat samples
    # .. in case of failure
    for repeat in range(2*nrepeat):
        desc = f"[{stationid}] Testing uniform sampling for "+\
                    f"marginal {marginal} ({repeat+1}/{nrepeat})"
        print(desc)

        # Generate parameters from prior
        try:
            dist.locn = norm.rvs(*ylocn_prior)
            dist.logscale = norm.rvs(*ylogscale_prior)
            dist.shape1 = norm.rvs(*yshape1_prior)
        except:
            continue

        # Generate data from prior params
        ysmp = dist.rvs(N)

        # Configure stan data and initialisation
        try:
            sv = sample.StanSamplingVariable(ysmp, marginal)
        except:
            continue

        sv.locn_prior = ylocn_prior
        sv.logscale_prior = ylogscale_prior
        sv.shape1_prior = yshape1_prior

        stan_data = sv.to_dict()
        stan_inits = sv.initial_parameters

        # Clean output folder
        fout = FTESTS / "sampling" / "univariate" / stationid / marginal
        fout.mkdir(parents=True, exist_ok=True)
        for f in fout.glob("*.*"):
            f.unlink()

        # Sample
        try:
            smp = univariate_censored_sampling(data=stan_data,
                                               inits=stan_inits,
                                               chains=4,
                                               seed=SEED,
                                               iter_warmup=5000,
                                               iter_sampling=500,
                                               output_dir=fout)
        except Exception as err:
            continue

        # Get sample data
        df = smp.draws_pd()

        # Test statistic
        Nsmp = len(df)
        test_stat.append([(df.loc[:, f"y{cn}"]<dist[cn]).sum()/Nsmp\
                for cn in parnames])

        # Stop iterating when the number of samples is met
        if repeat>nrepeat-2:
            break

    test_stat = pd.DataFrame(test_stat, columns=parnames)

    putils.ecdfplot(ax, test_stat)
    ax.axline((0, 0), slope=1, linestyle="--", lw=0.9)

    title = f"{marginal} - {len(test_stat)} samples / {nrepeat}"
    ax.set_title(title)

    fig.suptitle(f"Station {stationid}")
    fp = FTESTS / "images" / f"univariate_sampling_{stationid}_{marginal}.png"
    fp.parent.mkdir(exist_ok=True)
    fig.savefig(fp)

