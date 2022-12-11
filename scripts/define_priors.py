#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2022-12-11 Sun 04:13 PM
## Comment : Define priors
##
## ------------------------------

import sys, json, re, math
from pathlib import Path

import numpy as np
import pandas as pd

import warnings

from tqdm import tqdm

from nrivfloodfreqstan import marginals
from hydrodiy.io import iutils

import importlib
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

spec = importlib.util.spec_from_file_location("test_sample", froot / "tests" / "test_sample.py")
test_sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_sample)

#----------------------------------------------------------------------
# Configure
#----------------------------------------------------------------------

TQDM_DISABLE = True

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------

fout = froot / "outputs" / "priors"
fout.mkdir(exist_ok=True, parents=True)

basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
stationids = test_sample.get_stationids()
stations = test_sample.get_info()

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

for marginal in ["GEV", "LogPearson3", "Gumbel", "LogNormal"]:
    dist = marginals.factory(marginal)

    res = []
    for stationid in stationids:
        y = test_sample.get_ams(stationid)

        # Fit using lh moments
        for eta in [2, 1, 0]:
            try:
                dist.fit_lh_moments(y, eta)
                break
            except:
                pass

        # Store data
        dd = {\
            "stationid": stationid, \
            "area": stations.loc[stationid, "Catchment_Area"], \
            "lon": stations.loc[stationid, "Longitude"], \
            "lat": stations.loc[stationid, "Latitude"], \
            "locn": dist.locn, \
            "logscale": dist.logscale, \
            "shape1": dist.shape1
        }
        res.append(dd)

    res = pd.DataFrame(res)

    # Lin reg
    cpreds = ["area", "lon", "lat"]
    X = np.column_stack([np.ones(len(res)), res.loc[:, cpreds]])
    for pname in ["locn", "logscale", "shape1"]:
        y = res.loc[:, pname]
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        yhat = X.dot(theta)

        sys.exit()




LOGGER.info("Process completed")
