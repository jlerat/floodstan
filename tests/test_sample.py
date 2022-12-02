import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt

from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal

import pytest
import warnings

from cmdstanpy import CmdStanModel

import importlib
from tqdm import tqdm

from nrivfloodfreq import fdist, fsample
from nrivfloodfreqstan import sample, test_stan_functions
from hydrodiy.io import csv

#import data_reader

np.random.seed(5446)

FTESTS = Path(__file__).resolve().parent

# --- Utils functions ----------------------------
def get_stationids():
    fs = FTESTS / "data"
    stationids = []
    for f in fs.glob("*_AMS.csv"):
        stationids.append(re.sub("_.*", "", f.stem))
    return stationids

def get_ams(stationid):
    fs = FTESTS / "data" / f"{stationid}_AMS.csv"
    df = pd.read_csv(fs, skiprows=15, index_col=0)
    return df.iloc[:, 0]


# ------------------------------------------------

def test_get_stan_model():
    for modname in sample.STAN_MODEL_NAMES:
        model = sample.load_stan_model(modname)
        print(f"\n{modname}: {model.exe_info()}\n")


def test_get_copula_prior():
    prior = sample.get_copula_prior("uninformative")
    assert np.allclose(prior, [0.8, 1])


def test_get_marginal_prior():
    prior_variables = {"area": 100}
    prior = sample.get_marginal_prior("streamflow_obs", "GEV", \
                                prior_variables, "uninformative")

    expected = {\
        "loc": [700./3, 1400./3], \
        "logscale": [5, 4], \
        "shape": [0, 4]
    }

    for key in ["loc", "logscale", "shape"]:
        assert np.allclose(prior[key], expected[key])



def test_prepare():
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


def test_marginals():
    stationids = get_stationids()
    for stationid in stationids:
        z = get_ams(stationid)
        df = pd.DataFrame({"y": z, "z": z})
        yz = df.values
        LOGGER = sample.get_logger(level="INFO", stan_logger=False)

        zcensor = z.quantile(0.5)
        N = len(df)
        prior_variables = {"area": 500}

        for marginal in ["LogNormal", "Gumbel", "GEV", "LogPearson3", "Normal"]:
            LOGGER.info(f"[{stationid}] Testing marginal {marginal}")
            dist = fdist.factory(marginal)
            params = dist.fit_lh_moments(df.z).loc[0]
            dist.set_dict_params(params.to_dict())

            if marginal == "LogNormal":
                loc = dist.m
                logscale = math.log(dist.s)
                shape = 0
            elif marginal == "Normal":
                loc = dist.mu
                logscale = dist.logsig
                shape = 0
            elif marginal == "Gumbel":
                loc = dist.tau
                logscale = dist.logalpha
                shape = 0
            elif marginal == "GEV":
                loc = dist.tau
                logscale = dist.logalpha
                shape = dist.kappa
            elif marginal == "LogPearson3":
                loc = dist.s
                logscale = math.log(dist.s)
                shape = dist.g

            for i in range(N):
                stan_data = sample.prepare([yz[i, 0]], [yz[i, 1]], \
                                        ymarginal="Normal", \
                                        zmarginal=marginal, \
                                        zcensor=zcensor, \
                                        prior_variables=prior_variables)
                stan_data["yloc"] = 10
                stan_data["ylogscale"] = 0
                stan_data["yshape"] = 0
                stan_data["rho"] = 0.5

                # We use z variable because the test functions
                # calls lpdf and lcdf for z
                stan_data["zloc"] = loc
                stan_data["zlogscale"] = logscale
                stan_data["zshape"] = shape

                # Run stan
                smp = test_stan_functions.sample(data=stan_data, \
                                    chains=1, iter_warmup=0, iter_sampling=1, \
                                    fixed_param=True, show_progress=False)
                smp = smp.draws_pd().squeeze()

                # Test
                z = yz[i, 1]
                n31 = stan_data["Ncases"][2, 0]
                l31 = dist.logpdf(z)
                if n31>0:
                    assert np.isclose(smp.l31, l31)

                n32 = stan_data["Ncases"][2, 1]
                l32 = dist.logcdf(zcensor)
                if n32>0:
                    assert np.isclose(smp.l32, l32)


def test_copulas():
    stationids = get_stationids()
    for stationid in stationids:
        z = get_ams(stationid)
        e = np.random.uniform(0, 10, len(z))
        y = z+e
        y.iloc[np.random.choice(np.arange(len(z)), len(z)//3)] = np.nan
        df = pd.DataFrame({"y": y, "z": z})
        yz = df.values
        LOGGER = sample.get_logger(level="INFO", stan_logger=False)

        ycensor = y.quantile(0.5)
        zcensor = z.quantile(0.5)
        N = len(df)
        prior_variables = {"area": 500}
        ymarginal = "Normal"
        zmarginal = "Normal"

        #for copula in ["Gumbel", "Clayton", "Gaussian"]:
        for copula in ["Gaussian"]:
            LOGGER.info(f"[{stationid}] Testing copula {copula}")


            for i in range(N):
                stan_data = sample.prepare([yz[i, 0]], [yz[i, 1]], \
                                        ymarginal=ymarginal, \
                                        zmarginal=zmarginal, \
                                        ycensor=ycensor, \
                                        zcensor=zcensor, \
                                        copula=copula, \
                                        prior_variables=prior_variables)
                yloc, ylogscale = df.y.mean(), math.log(df.y.std())
                stan_data["yloc"] = yloc
                stan_data["ylogscale"] = ylogscale
                stan_data["yshape"] = 0

                zloc, zlogscale = df.z.mean(), math.log(df.z.std())
                stan_data["zloc"] = zloc
                stan_data["zlogscale"] = zlogscale
                stan_data["zshape"] = 0

                rho = np.random.uniform(-0.99, 0.99)
                stan_data["rho"] = rho

                # Run stan
                smp = test_stan_functions.sample(data=stan_data, \
                                    chains=1, iter_warmup=0, iter_sampling=1, \
                                    fixed_param=True, show_progress=False)
                smp = smp.draws_pd().squeeze()

                # Test
                cop = Copula(copula, rho)
                ucensor = norm.cdf(ycensor, loc=yloc, scale=math.exp(ylogscale))
                u = norm.cdf(stan_data["y"], loc=yloc, scale=math.exp(ylogscale))
                vcensor = norm.cdf(zcensor, loc=zloc, scale=math.exp(zlogscale))
                v = norm.cdf(stan_data["z"], loc=zloc, scale=math.exp(zlogscale))

                n11 = stan_data["Ncases"][0, 0]
                if n11>0:
                    pass
                #real l11_a = copula_lpdf(uv[i11,:] | copula, rho);

                n21 = stan_data["Ncases"][1, 0]
                if n21>0:
                    pass
                #real l21_a = copula_lpdf_ucensored(ucensor, uv[i21, 2], copula, rho);

                n12 = stan_data["Ncases"][0, 1]
                if n12>0:
                    pass
                #real l12_a = copula_lpdf_ucensored(vcensor, uv[i12, 1], copula, rho);


                n22 = stan_data["Ncases"][1, 1]
                if n22>0:
                    pass
                #real l22 = Ncases[2,2]*copula_lcdf(uvcensors | copula, rho);

