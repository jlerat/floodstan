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
from floodstan import report
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

@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
def test_stan_sampling_variable(distname, allclose):
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

    sv = sample.StanSamplingVariable(y, distname, censor)
    assert allclose(sv.censor, censor)
    assert isinstance(sv.marginal, marginals.FloodFreqDistribution)
    assert allclose(sv.data, y)
    assert sv.N == len(y)
    assert sv.marginal_code == marginals.MARGINAL_NAMES[distname]
    assert sv.marginal_name == distname
    nhigh, nlow = (y>=censor).sum(), (y<censor).sum()
    Ncases = [[nhigh, 0, 0], [nlow, 0, 0], [0, 0, 0]]
    assert allclose(sv.Ncases, Ncases)

    i11 = np.where(y>=censor)[0]+1
    assert allclose(sv.i11, i11)

    dd = sv.to_dict()
    keys = ["ymarginal", "y",
            "ycensor",
            "ylocn_prior", "ylogscale_prior",
            "yshape1_prior"]
    for key in keys:
        assert key in dd

    if distname != "LogPearson3":
        prior = [marginals.SHAPE1_PRIOR_LOC_DEFAULT,
                 marginals.SHAPE1_PRIOR_SCALE_DEFAULT]
        assert allclose(dd["yshape1_prior"], prior)
    else:
        assert allclose(dd["yshape1_prior"][1],
                        marginals.SHAPE1_PRIOR_SCALE_DEFAULT)

    # Rapid setting
    sv = sample.StanSamplingVariable(y)

    msg = "Initial parameters"
    with pytest.raises(ValueError, match=msg):
        ip = sv.initial_parameters

    sv = sample.StanSamplingVariable(y, distname)
    d = sv.data

    # Initial values
    inits = sv.initial_parameters
    for init in inits:
        for pn in ["locn", "logscale", "shape1"]:
            assert pn in init

    cdfs = sv.initial_cdfs
    assert len(cdfs) == len(inits)
    assert len(cdfs[0]) == len(sv.data)


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
def test_stan_sampling_dataset(distname, allclose):
    y = get_ams("203010")
    z = get_ams("201001")
    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    y, z = df.y, df.z

    yv = sample.StanSamplingVariable(y, distname, 100)
    zv = sample.StanSamplingVariable(z, distname, 100)
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


@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("censoring", [False, True])
def test_univariate_censored_sampling(distname, censoring, allclose):
    stationids = get_stationids()
    stationid = stationids[0]
    y = get_ams(stationid)
    censor = y.median() if censoring else np.nanmin(y) - 1.

    # Set STAN
    stan_nwarm = 10000
    stan_nsamples = 5000
    stan_nchains = 5

    sv = sample.StanSamplingVariable(y, distname, censor,
                                     ninits=stan_nchains)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters
    stan_inits3 = stan_inits[:3]

    # Clean output folder
    fout = FTESTS / "sampling" / "univariate_short_syntax"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Wrong number of inits
    msg = "Expected 1 or"
    with pytest.raises(ValueError, match=msg):
        smp = univariate_censored_sampling(data=stan_data,
                                       inits=stan_inits3,
                                       output_dir=fout)

    # Sample
    smp = univariate_censored_sampling(data=stan_data,
                                       chains=stan_nchains,
                                       seed=SEED,
                                       iter_warmup=stan_nwarm,
                                       iter_sampling=stan_nsamples // stan_nchains,
                                       inits=stan_inits,
                                       output_dir=fout)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    import pdb; pdb.set_trace()


    # Test diag
    assert diag["treedepth"] == "satisfactory"
    assert diag["ebfmi"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"

    # Test divergence
    prc = diag["divergence_proportion"]

    easy = ["GEV", "Normal", "Gamma", "LogNormal",
            "Gumbel"]
    moderate = ["GeneralizedLogistic"]
    hard = ["LogPearson3", "GeneralizedPareto"]

    if distname in easy:
        thresh = 5 if censoring else 1
    elif distname in moderate:
        thresh = 20 if censoring else 8
    else:
        # Stupid !!!
        thresh = 80 if censoring else 50

    assert prc < thresh
