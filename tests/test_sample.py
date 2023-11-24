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
from floodstan import test_marginal, test_copula, \
                                univariate_censoring, \
                                bivariate_censoring

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent

TQDM_DISABLE = True

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

    df.columns = [re.sub(" |,", "_", re.sub(" \(.*", "", cn)) \
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

    msg = "Data is not set."
    with pytest.raises(ValueError, match=msg):
        d = sv.data

    msg = "Cannot find"
    with pytest.raises(AssertionError, match=msg):
        sv.set(y, "bidule", 1)

    msg = "Expected data"
    with pytest.raises(AssertionError, match=msg):
        sv.set(y[:, None], "GEV", censor)

    sv.set(y, "GEV", censor)
    assert allclose(sv.censor, censor)

    sv.set(y, "GEV", 1.)
    assert allclose(sv.censor, max(y.min(), 1.))
    assert allclose(sv.data, y)
    assert sv.N == len(y)
    assert sv.marginal_code == 3
    assert sv.marginal_name == "GEV"
    assert allclose(sv.Ncases, [[55, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert allclose(sv.i11, np.arange(sv.N)+1)

    dd = sv.to_dict()
    assert "xmarginal" in dd
    assert "x" in dd
    assert "xcensor" in dd

    # Rapid setting
    sv = sample.StanSamplingVariable(y)
    msg = "Data is not set."
    with pytest.raises(ValueError, match=msg):
        d = sv.data

    sv = sample.StanSamplingVariable(y, "GEV")
    d = sv.data


def test_stan_sampling_variable_prior(allclose):
    y = get_ams("203010")

    sv = sample.StanSamplingVariable(y, "GEV")
    dd = sv.to_dict()
    assert not "xlocn_prior" in dd

    sp = sample.StanSamplingVariablePrior(sv)
    sp.add_priors(dd)
    assert "xlocn_prior" in dd


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
    assert "ymarginal" in dd
    assert "zmarginal" in dd

    i11 = dset.i11
    assert pd.notnull(df.y.iloc[i11-1]).all()
    assert pd.notnull(df.z.iloc[i11-1]).all()

    i31 = dset.i31
    assert pd.isnull(df.y.iloc[i31-1]).all()
    assert pd.notnull(df.z.iloc[i31-1]).all()


def test_marginals_vs_stan(allclose):
    stationids = get_stationids()
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    nboot = 100
    marginal_names = sample.MARGINAL_NAMES
    if TQDM_DISABLE:
        print("\n")

    for stationid in stationids:
        y = get_ams(stationid)
        N = len(y)

        for marginal in marginal_names:
            dist = marginals.factory(marginal)
            desc = f"[{stationid}] Testing stan {marginal} marginal"
            tbar = tqdm(range(nboot), desc=desc, disable=TQDM_DISABLE)
            if TQDM_DISABLE:
                print(desc)

            for iboot in tbar:
                # Bootstrap fit
                rng = np.random.default_rng(SEED)
                yboot = rng.choice(y.values, N)
                dist.params_guess(yboot)
                y0, y1 = dist.support

                sv = sample.StanSamplingVariable(yboot, marginal)
                sv.name = "y"
                stan_data = sv.to_dict()

                stan_data["ylocn"] = dist.locn
                stan_data["ylogscale"] = dist.logscale
                stan_data["yshape1"] = dist.shape1

                # Run stan
                smp = test_marginal.sample(data=stan_data, \
                                    chains=1, iter_warmup=0, iter_sampling=1, \
                                    fixed_param=True, show_progress=False)
                smp = smp.draws_pd().squeeze()

                # Test
                luncens = smp.filter(regex="luncens").values
                expected = dist.logpdf(yboot)
                assert allclose(luncens, expected, atol=1e-5)

                cens = smp.filter(regex="^cens").values
                expected = dist.cdf(yboot)
                assert allclose(cens, expected, atol=1e-5)

                lcens = smp.filter(regex="^lcens").values
                expected = dist.logcdf(yboot)
                assert allclose(lcens, expected, atol=1e-5)


def test_copulas_vs_stan(allclose):
    N = 500
    rng = np.random.default_rng(SEED)
    uv = rng.uniform(0, 1, size=(N, 2))
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    if TQDM_DISABLE:
        print("\n")

    for copula in ["Gumbel", "Clayton", "Gaussian"]:
        cop = copulas.factory(copula)

        rmin = cop.rho_min
        rmax = cop.rho_max
        nval = 20
        desc = "Testing stan copula "+copula
        tbar = tqdm(np.linspace(rmin, rmax, nval), \
                        desc=desc, disable=TQDM_DISABLE, total=nval)
        if TQDM_DISABLE:
            print(desc)

        for rho in tbar:
            cop.rho = rho

            stan_data = {
                "copula": sample.COPULA_NAMES[copula], \
                "N": N, \
                "uv": uv, \
                "rho": rho
            }

            # Run stan
            smp = test_copula.sample(data=stan_data, \
                                chains=1, iter_warmup=0, iter_sampling=1, \
                                fixed_param=True, show_progress=False)
            smp = smp.draws_pd().squeeze()

            assert allclose(smp.rho_check, rho, atol=1e-6)

            # Test copula pdf
            lpdf = smp.filter(regex="luncens")
            expected = cop.logpdf(uv)
            assert allclose(lpdf, expected, atol=1e-7)

            # test copula cdf
            lcdf = smp.filter(regex="lcens")
            expected = cop.logcdf(uv)
            assert allclose(lcdf, expected, atol=1e-7)

            # Test copula conditional density
            lcond = smp.filter(regex="lcond")
            expected = np.log(cop.conditional_density(uv[:, 0], uv[:, 1]))
            assert allclose(lcond, expected, atol=1e-7)


def test_univariate_sampling(allclose):
    # Testing univariate sampling following the process described by
    # Samantha R Cook, Andrew Gelman & Donald B Rubin (2006)
    # Validation of Software for Bayesian Models Using Posterior Quantiles,
    # Journal of Computational and Graphical Statistics, 15:3, 675-692,
    # DOI: 10.1198/106186006X136976

    stationids = get_stationids()
    stationids = stationids[:3]
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    plots = {i: n for i, n in enumerate(sample.MARGINAL_NAMES.keys())}

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 100
    nrepeat = 50
    nrows, ncols = 3, 3
    axwidth, axheight = 5, 5
    print("\n")

    for stationid in stationids:
        y = get_ams(stationid)
        N = len(y)

        # Setup image
        plt.close("all")
        mosaic = [[plots.get(ncols*ir+ic, ".") for ic in range(ncols)]\
                            for ir in range(nrows)]

        w, h = axwidth*ncols, axheight*nrows
        fig = plt.figure(figsize=(w, h), layout="tight")
        axs = fig.subplot_mosaic(mosaic)

        for marginal, ax in axs.items():
            if marginal in ["Gumbel", "LogNormal", "Normal", "Gamma"]:
                parnames = ["locn", "logscale"]
            else:
                parnames = ["locn", "logscale", "shape1"]

            dist = marginals.factory(marginal)
            dist.params_guess(y)

            if dist.shape1<marginals.SHAPE1_MIN or dist.shape1>marginals.SHAPE1_MAX:
                continue

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
                dist.locn = norm.rvs(*ylocn_prior)
                dist.logscale = norm.rvs(*ylogscale_prior)
                dist.shape1 = norm.rvs(*yshape1_prior)
                # Generate data from prior params
                ysmp = dist.rvs(N)

                # Configure stan data and initialisation
                sv = sample.StanSamplingVariable(ysmp, marginal)
                sv.name = "y"
                stan_data = sv.to_dict()

                stan_data["ylocn_prior"] = ylocn_prior
                stan_data["ylogscale_prior"] = ylogscale_prior
                stan_data["yshape1_prior"] = yshape1_prior

                # Clean output folder
                fout = FTESTS / "sampling" / "univariate" / stationid / marginal
                fout.mkdir(parents=True, exist_ok=True)
                for f in fout.glob("*.*"):
                    f.unlink()

                # Sample
                try:
                    smp = univariate_censoring.sample(\
                        data=stan_data, \
                        chains=4, \
                        seed=SEED, \
                        iter_warmup=5000, \
                        iter_sampling=500, \
                        output_dir=fout, \
                        inits=stan_data)
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
        fp = FTESTS / "images" / f"univariate_sampling_{stationid}.png"
        fp.parent.mkdir(exist_ok=True)
        fig.savefig(fp)


def test_bivariate_sampling(allclose):
    # Same background than univariate sampling tests
    return
    # TODO !

    stationids = get_stationids()
    nstations = 2
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    copula_names = sample.COPULA_NAMES
    plots = {i: n for i, n in enumerate(copula_names)}

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 100
    nrepeat = 50
    nrows, ncols = 2, 2
    axwidth, axheight = 5, 5

    for isite in range(nstations):
        # Create stan variables
        y = get_ams(stationids[isite])
        z = get_ams(stationids[isite+1])
        N = len(y)

        z.iloc[-2] = np.nan # to add a missing data in z
        df = pd.DataFrame({"y": y, "z": z}).sort_index()
        y, z = df.y, df.z

        yv = sample.StanSamplingVariable(y, "GEV", 100)
        zv = sample.StanSamplingVariable(z, "GEV", 100)

        # Setup image
        plt.close("all")
        mosaic = [[plots.get(ncols*ir+ic, ".") for ic in range(ncols)]\
                            for ir in range(nrows)]

        w, h = axwidth*ncols, axheight*nrows
        fig = plt.figure(figsize=(w, h), layout="tight")
        axs = fig.subplot_mosaic(mosaic)

        for cop in cops:
            pass

