import sys, re, math, json
from io import StringIO

import numpy as np
import pandas as pd

from pathlib import Path
import pytest
import warnings

FTESTS = Path(__file__).resolve().parent

STATIONS = ["203010", "203014", "arr61", \
             "arr84", "arr88", \
             "arr93", "LISAGGFLOW2"]
MODELS = ["bayesian-noprior", "LH0"]
SOURCES = ["flike", "bestfit"]
DISTRIBUTIONS = ["GEV", "LogNormal", "Gumbel", "LogPearson3"]

CENSORS = {
    "flike": {
        "203010": 382.52, \
        "203014": 251.83, \
        "arr61": 49.03, \
        "arr84": 54.4, \
        "arr88": 36.51, \
        "arr93": 39.1, \
        "LISAGGFLOW2": 505.6
    },
    "bestfit": {
        "203010": 382.52, \
        "203014": 222.39, \
        "arr61": 49.03, \
        "arr84": 54.4, \
        "arr88": 36.51, \
        "arr93": 39.1,\
        "LISAGGFLOW2": 505.6
    }
}

def read_flike_outputs(station, distname, model, \
                    censoring):
    txt = "" if censoring else "no"
    fr = FTESTS / "data" / "flike_outputs" / \
                    f"{station}_{distname}_{model}_{txt}censoring.txt"

    if not fr.exists():
        errmsg = f"Cannot find FLIKE data for {station}/"+\
                    f"{distname}/{model}/{txt}censoring"
        raise FileNotFoundError(errmsg)

    with fr.open("r") as fo:
        txt = fo.readlines()

    # Setup search engines
    names = ["data", "lmoments", "fit", "quantiles", "posterior", "maxpost"]
    specs = {n:{"txt": "", "delta": 2, "start": 1000, "active": False} for n in names}
    specs["data"]["search"] = "Rank|error zone"
    specs["lmoments"]["search"] = "L moment"
    specs["posterior"]["search"] = "Parameter.*Correlation"
    specs["fit"]["search"] = "Parameter.*(LH|Most probable)"
    specs["quantiles"]["search"] = "AEP 1 in Y Quantile|probability limits"
    specs["maxpost"]["search"] = "Maximized log-posterior density"
    specs["maxpost"]["delta"] = 0

    # Perform search line by line
    for iline, line in enumerate(txt):
        for name in names:
            spec = specs[name]
            if re.search(spec["search"], line):
                spec["start"] = iline+spec["delta"]

            if iline == spec["start"]:
                spec["active"] = True

            if line.strip() == "" or re.search(f"{'-'*20}", line):
                spec["active"] = False

            if spec["active"]:
                spec["txt"] += line

    # Convert txt to data frames for easy data use
    for name in names:
        txt = specs[name]["txt"]
        df = pd.read_fwf(StringIO(txt), header=None) if txt != "" else None
        if name == "data":
            try:
                df = df.astype(float)
            except:
                # Deal with spaces in columns
                df = df.apply(lambda x: x.str.replace(" .*", "").astype(float))
            cols = ["rank", "streamflow"]+[f"C{i}" for i in range(df.shape[1]-2)]
            df.columns = cols
        elif name == "quantiles":
            df.columns = ["AEP", "quantile", "5%", "95%"]+\
                            [f"C{i}" for i in range(df.shape[1]-4)]
            df = df.loc[df.AEP>=2, :]
            df.loc[:, "AEP"] = df.AEP.astype(int)
            df = df.set_index("AEP")

        specs[name]["df"] = df


    # Fix log transform of streamflow data
    # in certain result files
    df = specs["data"]["df"]
    if distname in ["LogNormal", "LogPearson3"] and model.startswith("LH"):
        logq10 = df.streamflow.values
        logq = logq10*math.log(10)
        df.loc[:, "streamflow_log"] = logq
        df.loc[:, "streamflow_log10"] = logq10
        df.loc[:, "streamflow"] = np.exp(logq)
    else:
        q = df.streamflow
        logq = np.log(q)
        df.loc[:, "streamflow_log"] = logq
        df.loc[:, "streamflow_log10"] = logq/math.log(10)

    dataframes = {n: specs[n]["df"] for n in names}
    return dataframes, fr


def read_bestfit_outputs(station, distname, model, \
                    censoring):
    txt = "" if censoring else "no"
    fp = FTESTS / "data" / "bestfit_outputs" / \
                    f"{station}_{distname}_{model}_{txt}censoring_parameters.txt"
    fq = FTESTS / "data" / "bestfit_outputs" / \
                    f"{station}_{distname}_{model}_{txt}censoring_quantiles.txt"

    if not fp.exists() or not fq.exists():
        errmsg = f"Cannot find BESFIT data for {station}/"+\
                    f"{distname}/{model}/{txt}censoring"
        raise FileNotFoundError(errmsg)

    # Format parameters
    params = pd.read_csv(fp, sep="\t").iloc[:, 1:-1]

    if distname == "GEV":
        params.columns = ["tau", "alpha", "kappa"]
    elif distname == "LogNormal":
        params.columns = ["m", "s"]
        params *= math.log(10)
    elif distname == "LogPearson3":
        params.columns = ["m", "s", "g"]
        params.loc[:, ["m", "s"]] *= math.log(10)
    elif distname == "Gumbel":
        params.columns = ["tau", "alpha"]

    params = params.describe().loc[["mean", "std"], :].T

    # Format quantiles
    quantiles = pd.read_csv(fq, sep="\t")
    quantiles.loc[:, "AEP"] = (1/quantiles.AEP).astype(int)
    idx = (quantiles.AEP!=3)&(quantiles.AEP>=2)
    quantiles = quantiles.loc[idx, :].set_index("AEP").iloc[:, :2]
    quantiles = quantiles.rename(columns={"5.0% CI": "5%", \
                                "95.0% CI": "95%"})
    quantiles = quantiles.loc[:, ["5%", "95%"]].round(1)

    testdata = {"params": params, "quantiles": quantiles}
    return testdata, fp



def get_test_data(station, distname, model, censoring, source):
    assert station in STATIONS
    assert distname in DISTRIBUTIONS
    assert model in MODELS
    assert source in SOURCES

    if source == "flike":
        return read_flike_outputs(station, distname, \
                                        model, censoring)
    else:
        return read_bestfit_outputs(station, distname, \
                                        model, censoring)
