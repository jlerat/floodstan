import sys, re, math, json
from pathlib import Path
import logging
import numbers

from datetime import datetime
import time

import numpy as np
import pandas as pd

from floodstan import marginals

MARGINAL_CODES = {
    "Gumbel": 1, \
    "LogNormal": 2,\
    "GEV": 3, \
    "LogPearson3": 4, \
    "Normal": 5, \
    "GeneralizedPareto": 6
}

COPULA_CODES = {
    "Gumbel": 1, \
    "Clayton": 2, \
    "Gaussian": 3
}

MARGINAL_CODES_INV = {code:name for name, code in MARGINAL_CODES.items()}
COPULA_CODES_INV = {code:name for name, code in COPULA_CODES.items()}


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


class StanData():
    def __init__(self):
        self._N = 0
        self._y = None
        self._z = None
        self._ymarginal = 3 # GEV
        self._zmarginal = 3 # GEV
        self._copula = 1 # Gumbel
        self._ycensor = 1e-10
        self._zcensor = 1e-10
        self._start = {}

    @property
    def N(self):
        return self._N

    @property
    def start(self):
        assert len(self._start)>0, "Start has not been set."
        return self._start

    @property
    def ymarginal_name(self):
        return MARGINAL_CODES_INV[self._ymarginal]

    @property
    def ymarginal(self):
        return self._ymarginal

    @property
    def zmarginal_name(self):
        return MARGINAL_CODES_INV[self._zmarginal]

    @property
    def zmarginal(self):
        return self._zmarginal

    @property
    def copula(self):
        return self._copula

    @copula.setter
    def copula(self, value):
        assert value in COPULA_CODES_INV, "Cannot find y copula code."
        self._copula = value

    @property
    def ycensor(self):
        return self._ycensor

    @property
    def y(self):
        if self._y is None:
            errmess = "y is not set."
            raise ValueError(errmess)
        return self._y

    @property
    def zcensor(self):
        return self._zcensor

    @property
    def z(self):
        if self._z is None:
            # Case where no z has been setup
            return 2*self.zcensor*np.ones(self.N)
        else:
            return self._z


    def set_y(self, y, ymarginal=3, ycensor=1e-10):
        assert ymarginal in MARGINAL_CODES_INV, "Cannot find y marginal code."
        self._ymarginal = ymarginal
        self._ycensor = np.float64(ycensor)

        y = np.array(values).astype(np.float64)
        assert y.ndim==1, "Expected y as 1d array."
        self._y = y
        self._N = len(y)

        # Set initial values
        dist = marginals.factory(self.ymarginal_name)
        dist.params_guess(y)
        # TODO verify init is compatible
        self._start["ylocn"] = dist.locn
        self._start["ylogscale"] = dist.logscale
        self._start["yshape1"] = dist.shape1

        # Set data indexes
        self.set_indexes()


    def set_z(self, z, zmarginal=3, zcensor=1e-10):
        assert zmarginal in MARGINAL_CODES_INV, "Cannot find z marginal code."
        self._zmarginal = zmarginal
        self._zcensor = np.float64(zcensor)

        z = np.array(values).astype(np.float64)
        assert z.ndim==1, "Expected z as 1d array."
        assert ~np.isnan(z).any(), "Expected no nan in z."
        assert len(z) == self.N, f"Expected z of length {N}."

        # Set initial values
        dist = marginals.factory(self.zmarginal_name)
        dist.params_guess(z)
        # TODO verify init is compatible
        self._start["zlocn"] = dist.locn
        self._start["zlogscale"] = dist.logscale
        self._start["zshape1"] = dist.shape1

        # Set data indexes
        self.set_indexes()


    def set_indexes(self):
        y, ycensor = self.y, self.ycensor
        ymiss = pd.isnull(y)
        yobs = y>=ycensor
        ycens = y<ycensor

        z, zcensor = self.z, self.zcensor
        zmiss = pd.isnull(z)
        zobs = z>=zcensor
        zcens = z<zcensor

        self.i11 = np.where(yobs & zobs)[0]+1
        self.i21 = np.where(ycens & zobs)[0]+1
        self.i31 = np.where(ymiss & zobs)[0]+1

        self.i12 = np.where(yobs & zcens)[0]+1
        self.i22 = np.where(ycens & zcens)[0]+1
        self.i32 = np.where(ymiss & zcens)[0]+1

        self.i13 = np.where(yobs & zmiss)[0]+1
        self.i23 = np.where(ycens & zmiss)[0]+1
        self.i33 = np.where(ymiss & zmiss)[0]+1

        self.Ncases = np.array([\
                        [len(self.i11), len(self.i12), len(self.i13)], \
                        [len(self.i21), len(self.i22), len(self.i23)], \
                        [len(self.i31), len(self.i32), len(self.i33)]
                    ])
        assert len(y) == np.sum(Ncases)


    def get_priors(self):
        start = self.start
        ylocstart = start["ylocn"]
        ylocprior = [ylocstart, 10*abs(ylocstart)]

        # TODO verify prior is compatible

        if "zlocn" in start:
            zlocstart = start["zlocn"]
            zlocprior = [zlocstart, 10*abs(zlocstart)]
        else:
            zlocprior = [0, 10]

        return ylocprior, zlocprior


    def to_dict(self):
        """ Export stan data to be used by stan program """
        yprior, zprior = self.get_priors()

        dd = {
            "ymarginal": self.ymarginal, \
            "zmarginal": self.zmarginal, \
            "copula": COPULA_CODES[copula], \
            "N": self.N, \
            "y": self.y, \
            "z": self.z, \
            "ycensor": self.ycensor, \
            "zcensor": self.zcensor, \
            "Ncases": self.Ncases, \
            "i11": self.i11, \
            "i21": self.i21, \
            "i31": self.i31, \
            "i12": self.i12, \
            "i22": self.i22, \
            "i32": self.i32, \
            "i13": self.i13, \
            "i23": self.i23, \
            "i33": self.i33, \
            "logscale_lower": LOGSCALE_LOWER, \
            "logscale_upper": LOGSCALE_UPPER, \
            "shape1_lower": marginals.SHAPE1_MIN, \
            "shape1_upper": marginals.SHAPE1_MAX, \
            "rho_lower": RHO_LOWER, \
            "rho_upper": RHO_UPPER, \
            "rho_prior": [0.8, 1], \
            "ylocn_prior": yprior, \
            "ylogscale_prior": [0, 10], \
            "yshape1_prior": [0, 4], \
            "zlocn_prior": zprior, \
            "zlogscale_prior": [0, 10], \
            "zshape1_prior": [0, 4], \
        }
        return dd



