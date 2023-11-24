import sys, re, math, json
from pathlib import Path
import logging
import numbers

from datetime import datetime
import time

import numpy as np
import pandas as pd

from floodstan import marginals
from floodstan.marginals import MARGINAL_NAMES

COPULA_NAMES = {
    "Gumbel": 1, \
    "Clayton": 2, \
    "Gaussian": 3
}

MARGINAL_CODES = {code:name for name, code in MARGINAL_NAMES.items()}
COPULA_CODES = {code:name for name, code in COPULA_NAMES.items()}

# BOUNDS
LOGSCALE_LOWER = -5
LOGSCALE_UPPER = 10

RHO_LOWER = 0.01
RHO_UPPER = 0.95

# PRIOR
RHO_PRIOR = [0.8, 1]

# Logging
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



class StanSamplingVariable():
    def __init__(self, data=None, marginal_name=None, censor=1e-10):
        self.name = "x"
        self._N = 0
        self._data = None
        self._marginal_code = None
        self._marginal_name = None
        self._censor = censor
        self._initial_parameters = {}
        self._is_miss = None
        self._is_obs = None
        self._is_cens = None

        # Set if the 3 inputs are set
        if not data is None and not marginal_name is None\
                        and not censor is None:
            self.set(data, marginal_name, censor)

    @property
    def N(self):
        return self._N

    @property
    def initial_parameters(self):
        assert len(self._initial_parameters)>0, \
                f"Variable {self.name}: Initial parameters have not been set."
        return self._initial_parameters

    @property
    def marginal_name(self):
        return self._marginal_name

    @property
    def marginal_code(self):
        return MARGINAL_NAMES[self._marginal_name]

    @property
    def censor(self):
        return self._censor

    @property
    def is_miss(self):
        return self._is_miss

    @property
    def is_obs(self):
        return self._is_obs

    @property
    def is_cens(self):
        return self._is_cens

    @property
    def data(self):
        if self._data is None:
            errmess = "Data is not set."
            raise ValueError(errmess)
        return self._data

    def set(self, data, marginal_name, censor):
        assert marginal_name in MARGINAL_NAMES, f"Cannot find marginal {marginal_name}."
        self._marginal_name = marginal_name

        data = np.array(data).astype(np.float64)
        assert data.ndim==1, "Expected data as 1d array."
        assert (~np.isnan(data)).sum()>5, "Need more than 5 valid data points."
        self._data = data
        self._N = len(data)

        dok = data[~np.isnan(data)]
        assert len(dok)>0, "Expected at least one valid data value."

        # We set the censor close to data min to avoid potential
        # problems with computing log cdf for the censor
        censor = max(np.float64(censor), dok.min()-1e-10)
        self._censor = censor

        # Set initial values
        dist = marginals.factory(self.marginal_name)
        dist.params_guess(dok)
        # .. verify parameter is compatible
        lpdf = dist.logpdf(dok)
        lcdf = dist.logcdf(censor)
        errmsg = "Initial parameters incompatible with data"
        assert not np.isnan(lpdf).any() or np.isnan(lcdf), errmsg

        self._initial_parameters["locn"] = dist.locn
        self._initial_parameters["logscale"] = dist.logscale
        self._initial_parameters["shape1"] = dist.shape1

        # Set indexes
        self._is_miss = pd.isnull(data)
        self._is_obs = data>=self.censor
        self._is_cens = data<self.censor
        self.i11 = np.where(self.is_obs)[0]+1
        self.i21 = np.where(self.is_cens)[0]+1
        # .. 3x3 due to stan code requirement. Only first 2 top left cells used
        self.Ncases = np.zeros((3, 3), dtype=int)
        self.Ncases[:2, 0] = [len(self.i11), len(self.i21)]


    def to_dict(self):
        """ Export stan data to be used by stan program """
        vn = self.name
        dd = {
            f"{vn}marginal": self.marginal_code, \
            "N": self.N, \
            vn: self.data, \
            f"{vn}censor": self.censor, \
            "logscale_lower": LOGSCALE_LOWER, \
            "logscale_upper": LOGSCALE_UPPER, \
            "shape1_lower": marginals.SHAPE1_MIN, \
            "shape1_upper": marginals.SHAPE1_MAX, \
            "i11": self.i11, \
            "i21": self.i21, \
            "Ncases": self.Ncases
        }
        for k, v in self.initial_parameters.items():
            dd[f"{vn}{k}"] = v

        return dd



class StanSamplingVariablePrior():
    def __init__(self, stanvar):
        self.stanvar = stanvar

    def add_priors(self, info):
        """ Add prior info to a dict """
        vn = self.stanvar.name
        start = self.stanvar.initial_parameters
        locn_start = start["locn"]
        info[f"{vn}locn_prior"] = [locn_start, 10*abs(locn_start)]

        logscale_start = start["logscale"]
        dscale = (LOGSCALE_UPPER-LOGSCALE_LOWER)/2
        info[f"{vn}logscale_prior"] = [logscale_start, dscale]

        shape1_start = start["shape1"]
        dshape = marginals.SHAPE1_MAX-marginals.SHAPE1_MIN
        info[f"{vn}shape1_prior"] = [shape1_start, dshape]



class StanSamplingDataset():
    def __init__(self, stan_variables, copula_name):
        # Set stan variables
        assert len(stan_variables)==2, "Only bivariate case accepted."
        # .. changing names to y and z (Stan code requirement)
        stan_variables[0].name = "y"
        stan_variables[1].name = "z"
        assert len(set([sv.N for sv in stan_variables]))==1, "Differing data size"
        self._stan_variables = stan_variables

        # Set copula
        assert copula_name in COPULA_NAMES, f"Cannot find copula {copula_name}."
        self._copula_name = copula_name

        # Set indexes
        self.set_indexes()

    @property
    def copula_name(self):
        return self._copula_name

    @property
    def copula_code(self):
        return COPULA_NAMES[self._copula_name]

    @property
    def stan_variables(self):
        return self._stan_variables

    def set_indexes(self):
        yv = self.stan_variables[0]
        zv = self.stan_variables[1]
        N = yv.N

        yobs = yv.is_obs
        ycens = yv.is_cens
        ymiss = yv.is_miss

        zobs = zv.is_obs
        zcens = zv.is_cens
        zmiss = zv.is_miss

        self.i11 = np.where(yobs & zobs)[0]+1
        self.i21 = np.where(ycens & zobs)[0]+1
        self.i31 = np.where(ymiss & zobs)[0]+1

        self.i12 = np.where(yobs & zcens)[0]+1
        self.i22 = np.where(ycens & zcens)[0]+1
        self.i32 = np.where(ymiss & zcens)[0]+1

        self.i13 = np.where(yobs & zmiss)[0]+1
        self.i23 = np.where(ycens & zmiss)[0]+1
        self.i33 = np.where(ymiss & zmiss)[0]+1
        assert len(self.i33) == 0, "Expected at least one variable to be valid."

        self.Ncases = np.array([\
                        [len(self.i11), len(self.i12), len(self.i13)], \
                        [len(self.i21), len(self.i22), len(self.i23)], \
                        [len(self.i31), len(self.i32), len(self.i33)]
                    ])
        assert N == np.sum(self.Ncases)

    def to_dict(self):
        dd = self.stan_variables[0].to_dict()
        dd.update(self.stan_variables[1].to_dict())
        dd.update({
            "copula": self.copula_code, \
            "rho_lower": RHO_LOWER, \
            "rho_upper": RHO_UPPER, \
            "rho_prior": RHO_PRIOR, \
            "Ncases": self.Ncases, \
            "i11": self.i11, \
            "i21": self.i21, \
            "i31": self.i31, \
            "i12": self.i12, \
            "i22": self.i22, \
            "i32": self.i32, \
            "i13": self.i13, \
            "i23": self.i23
        })
        return dd


