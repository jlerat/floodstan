import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal
from scipy.stats import ttest_1samp

import pytest
import warnings

from cmdstanpy import CmdStanModel

import importlib
from tqdm import tqdm

from nrivfloodfreqstan import marginals, sample, copulas
from nrivfloodfreqstan import test_marginal, test_copula, \
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

def test_get_copula_prior():
    prior = sample.get_copula_prior("uninformative")
    assert np.allclose(prior, [0.8, 1])


def test_get_marginal_prior():
    prior_variables = {"area": 100}
    prior = sample.get_marginal_prior("streamflow_obs", "GEV", \
                                prior_variables, "uninformative")

    expected = {\
        "locn": [700./3, 1400./3], \
        "logscale": [5, 4], \
        "shape1": [0, 4]
    }

    for key in ["locn", "logscale", "shape1"]:
        assert np.allclose(prior[key], expected[key])



def test_sample_prepare():
    y = get_ams("203010")
    z = get_ams("203014")
    df = pd.DataFrame({"y": y, "z": z})
    prior_variables = {"area": 500}
    stan_data = sample.prepare(df.y, df.z, prior_variables=prior_variables)

    Ncases = stan_data["Ncases"]
    assert np.allclose(Ncases, [[55, 0], [0, 0], [10, 0]])

    i11 = stan_data["i11"]
    assert pd.notnull(df.y.iloc[i11-1]).all()
    assert pd.notnull(df.z.iloc[i11-1]).all()

    i31 = stan_data["i31"]
    assert pd.isnull(df.y.iloc[i31-1]).all()
    assert pd.notnull(df.z.iloc[i31-1]).all()


def test_marginals(allclose):
    stationids = get_stationids()
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    nboot = 100

    for stationid in stationids:
        y = get_ams(stationid)
        N = len(y)

        for marginal in ["LogNormal", "Gumbel", "GEV", "LogPearson3", "Normal"]:
            dist = marginals.factory(marginal)
            desc = f"[{stationid}] Testing stan {marginal} marginal"
            tbar = tqdm(range(nboot), desc=desc, disable=TQDM_DISABLE)
            if TQDM_DISABLE:
                print("\n"+desc)

            for iboot in tbar:
                # Bootstrap fit
                rng = np.random.default_rng(SEED)
                yboot = rng.choice(y.values, N)
                dist.fit_lh_moments(yboot)
                y0, y1 = dist.support
                if y0>yboot.min() or y1<yboot.max():
                    continue

                stan_data = {
                    "ymarginal": sample.MARGINAL_CODES[marginal], \
                    "N": N, \
                    "y": yboot, \
                    "ylocn": dist.locn, \
                    "ylogscale": dist.logscale, \
                    "yshape1": dist.shape1
                }

                # Run stan
                smp = test_marginal.sample(data=stan_data, \
                                    chains=1, iter_warmup=0, iter_sampling=1, \
                                    fixed_param=True, show_progress=False)
                smp = smp.draws_pd().squeeze()

                # Test
                luncens = smp.filter(regex="luncens").values
                expected = dist.logpdf(yboot)
                assert allclose(luncens, expected, atol=1e-5)

                lcens = smp.filter(regex="lcens").values
                expected = dist.logcdf(yboot)
                assert allclose(lcens, expected, atol=1e-5)


def test_copulas(allclose):
    N = 100
    rng = np.random.default_rng(SEED)
    uv = rng.uniform(0, 1, size=(N, 2))
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    for copula in ["Gumbel", "Clayton", "Gaussian"]:
        cop = copulas.factory(copula)

        rmin = cop.rho_min
        rmax = cop.rho_max
        nval = 20
        desc = "Testing stan copula "+copula
        tbar = tqdm(np.linspace(rmin, rmax, nval), \
                        desc=desc, disable=TQDM_DISABLE, total=nval)
        if TQDM_DISABLE:
            print("\n"+desc)

        for rho in tbar:
            cop.rho = rho

            stan_data = {
                "copula": sample.COPULA_CODES[copula], \
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


def test_univariate(allclose):
    stationids = get_stationids()
    LOGGER = sample.get_logger(level="INFO")#, stan_logger=False)

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 1000

    for stationid in stationids:
        y = get_ams(stationid)
        N = len(y)

        for marginal in ["LogPearson3", "Gumbel", "LogNormal"]:
            dist = marginals.factory(marginal)
            dist.params_guess(y)

            if dist.shape1<marginals.SHAPE1_MIN or dist.shape1>marginals.SHAPE1_MAX:
                continue

            # Here we generate large number of data from known distribution
            ys = dist.rvs(nvalues)

            # Configure stan data and initialisation
            stan_data = sample.prepare(ys, ymarginal=marginal)

            # Flat priors
            dist2 = marginals.factory(marginal)
            dist2.params_guess(ys)
            stan_data["ylocn_prior"] = [dist2.locn, dist2.locn*10]
            stan_data["ylogscale_prior"] = [dist2.logscale, 5]

            # Initialise
            inits = sample.initialise(stan_data)

            # Clean output folder
            fout = FTESTS / "sampling" / "univariate" / stationid / marginal
            fout.mkdir(parents=True, exist_ok=True)
            for f in fout.glob("*.*"):
                f.unlink()

            # Sample
            smp = univariate_censoring.sample(\
                    data=stan_data, \
                    chains=4, \
                    seed=SEED, \
                    iter_warmup=10000, \
                    iter_sampling=5000, \
                    output_dir=fout, \
                    inits=inits)

            # Get sample data
            df = smp.draws_pd()
            diag = smp.diagnose()

            # T test on parameter samples
            for ip, pname in enumerate(["locn", "logscale", "shape1"]):
                ref = dist[pname]
                smp = df.loc[:, f"y{pname}"]

                # Does not work...
                st, pv = ttest_1samp(smp, ref)

                import pdb; pdb.set_trace()


