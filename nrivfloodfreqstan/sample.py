import sys, re, math, json
from pathlib import Path
import logging
import numbers

from datetime import datetime
import time

import numpy as np
import pandas as pd

from nrivfloodfreqstan import marginals

MARGINAL_CODES = {"Gumbel": 1, \
                    "LogNormal": 2,\
                    "GEV": 3, \
                    "LogPearson3": 4, \
                    "Normal": 5}

COPULA_CODES = {"Gumbel": 1, \
                "Clayton": 2, \
                "Gaussian": 3}

# BOUNDS
LOGSCALE_LOWER = -5
LOGSCALE_UPPER = 10

RHO_LOWER = 0.01
RHO_UPPER = 0.95

# Path to priors
FPRIORS = Path(__file__).resolve().parent / "priors"

LOGGER_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

def get_logger(level, flog=None, stan_logger=True):
    """ Get logger object.

    Parameters
    ----------
    level : str
        Logger level. See https://docs.python.org/3/howto/logging.html#logging-levels
    flog : str or Path
        Path to log file
    stan_logger : bool
        Use stan logger or not (to remove all stan messages)
    """
    if stan_logger:
        LOGGER = logging.getLogger("cmdstanpy")
    else:
        STAN_LOGGER = logging.getLogger("cmdstanpy")
        STAN_LOGGER.disabled = True
        LOGGER = logging.getLogger(Path(__file__).resolve().stem)

    # Set logging level
    if not level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"{level} not a valid level")

    LOGGER.setLevel(getattr(logging, level))

    # Set logging format
    ft = logging.Formatter(LOGGER_FORMAT)

    # log to console
    LOGGER.handlers = []
    if flog is None:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(ft)
        LOGGER.addHandler(sh)
    else:
        fh = logging.FileHandler(flog)
        fh.setFormatter(ft)
        LOGGER.addHandler(fh)

    return LOGGER


def get_copula_prior(prior_name="uninformative"):
    fp = FPRIORS / f"priors_{prior_name}.json"
    with fp.open("r") as fo:
        js = json.load(fo)

    return js["copula"]["rho"]


def get_marginal_prior(varname, marginal, \
                prior_variables={}, prior_name="uninformative"):
    fp = FPRIORS / f"priors_{prior_name}.json"
    with fp.open("r") as fo:
        js = json.load(fo)

    # Retrieve priors
    priors = js["marginal"][varname][marginal]

    # Evaluate
    for name, values in priors.items():
        values_ok = []
        for value in values:
            if isinstance(value, numbers.Number):
                values_ok.append(value)
            else:
                converted = eval(value.format(**prior_variables))
                values_ok.append(converted)

        priors[name] = values_ok

    return priors


def prepare(y, z=None, \
            ymarginal="GEV", \
            zmarginal="GEV", \
            yname="streamflow_obs", \
            zname="streamflow_awra", \
            copula="Gumbel", \
            ycensor=1e-10, \
            zcensor=1e-10, \
            prior_name="uninformative", \
            prior_variables={"area": 500}
        ):

    # Check inputs
    y = np.array(y).astype(np.float64)
    assert y.ndim==1, "Expected 1d array for y."

    if z is None:
        # All z data are uncensored
        z = np.zeros_like(y)+2*zcensor

    z = np.array(z).astype(np.float64)
    assert z.ndim==1, "Expected 1d array for z."
    N = len(z)
    assert len(y)==N, "Expected len(y)==len(z)."
    assert ~np.isnan(z).any(), "Expected no nan in z."

    ycensor = float(ycensor)
    zcensor = float(zcensor)

    # indexes
    ymiss = pd.isnull(y)
    yobs = y>=ycensor
    ycens = y<ycensor

    zmiss = pd.isnull(z)
    zobs = z>=zcensor
    zcens = z<zcensor

    i11 = np.where(yobs & zobs)[0]+1
    i21 = np.where(ycens & zobs)[0]+1
    i31 = np.where(ymiss & zobs)[0]+1

    i12 = np.where(yobs & zcens)[0]+1
    i22 = np.where(ycens & zcens)[0]+1
    i32 = np.where(ymiss & zcens)[0]+1

    i13 = np.where(yobs & zmiss)[0]+1
    i23 = np.where(ycens & zmiss)[0]+1
    i33 = np.where(ymiss & zmiss)[0]+1

    Ncases = np.array([[len(i11), len(i12), len(i13)], \
                [len(i21), len(i22), len(i23)], \
                [len(i31), len(i32), len(i33)]])
    assert N == np.sum(Ncases)

    # Create data dict
    stan_data = {\
        "ymarginal": MARGINAL_CODES[ymarginal], \
        "zmarginal": MARGINAL_CODES[zmarginal], \
        "copula": COPULA_CODES[copula], \
        "N": N, \
        "y": y, \
        "z": z, \
        "ycensor": ycensor, \
        "zcensor": zcensor, \
        "Ncases": Ncases, \
        "i11": i11, \
        "i21": i21, \
        "i31": i31, \
        "i12": i12, \
        "i22": i22, \
        "i32": i32, \
        "i13": i13, \
        "i23": i23, \
        "i33": i33, \
        "logscale_lower": LOGSCALE_LOWER, \
        "logscale_upper": LOGSCALE_UPPER, \
        "shape1_lower": marginals.SHAPE1_MIN, \
        "shape1_upper": marginals.SHAPE1_MAX, \
        "rho_lower": RHO_LOWER, \
        "rho_upper": RHO_UPPER
    }

    # Get priors
    copula_prior = get_copula_prior(prior_name)
    stan_data["rho_prior"] = copula_prior

    yprior = get_marginal_prior(yname, ymarginal, prior_variables, \
                                prior_name)
    stan_data["ylocn_prior"] = yprior["locn"]
    stan_data["ylogscale_prior"] = yprior["logscale"]
    stan_data["yshape1_prior"] = yprior["shape1"]

    zprior = get_marginal_prior(zname, zmarginal, prior_variables, \
                                prior_name)
    stan_data["zlocn_prior"] = zprior["locn"]
    stan_data["zlogscale_prior"] = zprior["logscale"]
    stan_data["zshape1_prior"] = zprior["shape1"]

    return stan_data


def initialise(stan_data):
    """ Initialise sampler.

    """
    # Get marginal distributions
    ymarginal = [k for k, code in MARGINAL_CODES.items()\
                        if code==stan_data["ymarginal"]][0]
    ydist = marginals.factory(ymarginal)
    ydist.params_guess(stan_data["y"])

    inits = {
        "ylocn": ydist.locn, \
        "ylogscale": ydist.logscale, \
        "yshape1": ydist.shape1
    }

    # Initialise covariate distribution
    # if there are valid covariate data
    if stan_data["z"].var()>0:
        zmarginal = [k for k, code in MARGINAL_CODES.items()\
                        if code==stan_data["zmarginal"]][0]
        zdist = marginals.factory(zmarginal)
        zdist.params_guess(stan_data["z"])

        inits["zlocn"] = zdist.locn
        inits["zlogscale"] = zdist.logscale
        inits["zshape1"] = zdist.shape1

    return inits


