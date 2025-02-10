import sys
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from floodstan import marginals

from floodstan import NCHAINS_DEFAULT

from floodstan.copulas import COPULA_NAMES
from floodstan.copulas import factory

from floodstan.marginals import MARGINAL_NAMES
from floodstan.marginals import LOGSCALE_LOWER
from floodstan.marginals import LOGSCALE_UPPER
from floodstan.marginals import SHAPE1_LOWER
from floodstan.marginals import SHAPE1_UPPER

from floodstan.discretes import DISCRETE_NAMES
from floodstan.discretes import PHI_LOWER
from floodstan.discretes import PHI_UPPER
from floodstan.discretes import LOCN_UPPER
from floodstan.discretes import NEVENT_UPPER


MARGINAL_CODES = {code: name for name, code in MARGINAL_NAMES.items()}
COPULA_CODES = {code: name for name, code in COPULA_NAMES.items()}
DISCRETE_CODES = {code: name for name, code in DISCRETE_NAMES.items()}

# Subset of copula currently supported in the stan model
COPULA_NAMES_STAN = ["Gaussian", "Clayton", "Gumbel"]

# Tight prior on shape parameter
SHAPE1_PRIOR = [0, 0.2]

# Prior on copula parameter
RHO_PRIOR = [0.7, 1.]

# Prior on discrete parameters
DISCRETE_LOCN_PRIOR = [1, 10]
DISCRETE_PHI_PRIOR = [1, 10]

CENSOR_DEFAULT = -1e10

# Logging
LOGGER_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOGGER_DATE_FORMAT = "%y-%m-%d %H:%M"


def get_logger(level="INFO", flog=None, stan_logger=True):
    """ Get logger object.

    Parameters
    ----------
    level : str
        Logger level. See
        https://docs.python.org/3/howto/logging.html#logging-levels
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
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"{level} not a valid level")

    LOGGER.setLevel(getattr(logging, level))

    # Set logging format
    ft = logging.Formatter(LOGGER_FORMAT, LOGGER_DATE_FORMAT)

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


def format_prior(values):
    values = np.array(values).astype(float)
    if values.shape != (2, ):
        errmess = f"Expected an array of shape (2, ), got {values.shape}."
        raise ValueError(errmess)
    return values


class StanSamplingVariable():
    def __init__(self,
                 data=None,
                 marginal_name=None,
                 censor=CENSOR_DEFAULT,
                 name="y",
                 ninits=NCHAINS_DEFAULT,
                 init_perturb_scale=0.1):
        self.name = str(name)
        if len(self.name) != 1:
            errmess = "Expected one character for name."
            raise ValueError(errmess)

        self._N = 0
        self._data = None
        self._marginal_code = None
        self._marginal_name = None
        self._marginal = None
        self._censor = float(censor)
        self._guess_parameters = []
        self._initial_parameters = []
        self._is_miss = None
        self._is_obs = None
        self._is_cens = None

        # Initial parameters
        self.ninits = ninits
        self.init_perturb_scale = init_perturb_scale
        self._initial_parameters = []
        self._initial_cdfs = []

        data_set = False
        if data is not None and censor is not None:
            self.set_data(data, censor)
            data_set = True

        marginal_set = False
        if marginal_name is not None:
            self.set_marginal(marginal_name)
            marginal_set = True

        if data_set and marginal_set:
            self.set_guess_parameters()
            self.set_initial_parameters()
            self.set_priors()

    @property
    def N(self):
        return self._N

    @property
    def guess_parameters(self):
        if len(self._guess_parameters) == 0:
            errmess = "Guess parameters have not been set."
            raise ValueError(errmess)

        return self._guess_parameters

    @property
    def initial_parameters(self):
        if len(self._initial_parameters) == 0:
            errmess = "Initial parameters have not been set."
            raise ValueError(errmess)

        return self._initial_parameters

    @property
    def initial_cdfs(self):
        if len(self._initial_cdfs) == 0:
            errmess = "Initial cdfs have not been set."
            raise ValueError(errmess)

        return self._initial_cdfs

    @property
    def marginal(self):
        if self._marginal is None:
            errmess = "Marginal has not been set."
            raise ValueError(errmess)
        return self._marginal

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
            errmess = "Data has not been set."
            raise ValueError(errmess)
        return self._data

    @property
    def locn_prior(self):
        if self._locn_prior is None:
            errmess = "locn_prior has not been set."
            raise ValueError(errmess)
        return self._locn_prior

    @locn_prior.setter
    def locn_prior(self, values):
        values = format_prior(values)
        self._locn_prior = values

    @property
    def logscale_prior(self):
        if self._logscale_prior is None:
            errmess = "logscale_prior has not been set."
            raise ValueError(errmess)
        return self._logscale_prior

    @logscale_prior.setter
    def logscale_prior(self, values):
        values = format_prior(values)
        self._logscale_prior = values

    @property
    def shape1_prior(self):
        if self._shape1_prior is None:
            errmess = "shape1_prior has not been set."
            raise ValueError(errmess)
        return self._shape1_prior

    @shape1_prior.setter
    def shape1_prior(self, values):
        values = format_prior(values)
        self._shape1_prior = values

    def set_marginal(self, marginal_name):
        if marginal_name not in MARGINAL_NAMES:
            errmess = f"Cannot find marginal {marginal_name}."
            raise ValueError(errmess)

        self._marginal_name = marginal_name

        dist = marginals.factory(self.marginal_name)
        self._marginal = dist

    def set_data(self, data, censor):
        data = np.array(data).astype(np.float64)

        if data.ndim != 1:
            errmess = "Expected data as 1d array."
            raise ValueError(errmess)

        if (~np.isnan(data)).sum() < 5:
            errmess = "Need more than 5 valid data points."
            raise ValueError(errmess)

        self._data = data
        self._N = len(data)

        dok = data[~np.isnan(data)]
        if len(dok) == 0:
            errmess = "Expected at least one valid data value."
            raise ValueError(errmess)

        # Set indexes
        self._is_miss = pd.isnull(data)
        self._is_obs = data >= self.censor
        self._is_cens = data < self.censor
        self.i11 = np.where(self.is_obs)[0] + 1
        self.i21 = np.where(self.is_cens)[0] + 1
        # .. 3x3 due to stan code requirement. Only first 2 top left cells used
        self.Ncases = np.zeros((3, 3), dtype=int)
        self.Ncases[:2, 0] = [len(self.i11), len(self.i21)]

        # We set the censor close to data min to avoid potential
        # problems with computing log cdf for the censor
        censor = max(np.float64(censor), dok.min() - 1e-10)
        self._censor = censor

    def set_guess_parameters(self):
        dok = self.data[~np.isnan(self.data)]
        dist = self.marginal

        # Get guess parameter set
        dist.params_guess(dok)
        params0 = dist.params
        self._guess_parameters = {
                "locn": params0[0],
                "logscale": params0[1],
                "shape1": params0[2]
                }

    def set_initial_parameters(self):
        data = self.data
        notnan = ~np.isnan(data)
        dist = self.marginal
        gp = self.guess_parameters
        params0 = np.array([gp["locn"], gp["logscale"],
                            gp["shape1"]])

        # Create a random sample for each chain
        inits = []
        while len(inits) < self.ninits:
            perturb = np.random.normal(loc=0,
                                       scale=self.init_perturb_scale,
                                       size=3)
            params = params0 + perturb
            params[0] = params0[0]*max(5e-2, 1+perturb[0])
            dist.params = params

            # Shift in scale to ensure valid start
            niter = 0
            while True and niter < 10:
                cdf_data = dist.cdf(data)
                cdf_data_min = cdf_data[notnan].min()
                cdf_data_max = cdf_data[notnan].max()
                cdf_censor = dist.cdf(self.censor)
                notok = (cdf_data_min < 1e-2) | (cdf_data_max > 0.99)
                notok |= (cdf_censor < 1e-2) | (cdf_censor > 0.99)
                if notok:
                    dist.logscale += 0.2
                    valid = False
                    niter += 1
                else:
                    valid = True
                    break

            if valid:
                # Parameter is valid, we can store it
                inits.append({
                    "locn": dist.locn,
                    "logscale": dist.logscale,
                    "shape1": dist.shape1
                    })
                # .. and the data cdf
                self._initial_cdfs.append(cdf_data)
            else:
                self._initial_parameters = []
                self._initial_cdfs = []
                errmess = "Cannot find initial marginal parameter."
                raise ValueError(errmess)

        self._initial_parameters = inits

    def set_priors(self):
        start = self.guess_parameters
        locn_start = start["locn"]
        self._locn_prior = [locn_start, 10 * abs(locn_start)]

        logscale_start = start["logscale"]
        dscale = (LOGSCALE_UPPER-LOGSCALE_LOWER) / 2
        self._logscale_prior = [logscale_start, dscale]

        self._shape1_prior = SHAPE1_PRIOR

    def to_dict(self):
        """ Export stan data to be used by stan program """
        vn = self.name
        dd = {
            f"{vn}marginal": self.marginal_code,
            "N": self.N,
            vn: self.data,
            f"{vn}censor": self.censor,
            f"{vn}locn_prior": self.locn_prior,
            f"{vn}logscale_prior": self.logscale_prior,
            f"{vn}shape1_prior": self.shape1_prior,
            "logscale_lower": LOGSCALE_LOWER,
            "logscale_upper": LOGSCALE_UPPER,
            "shape1_lower": SHAPE1_LOWER,
            "shape1_upper": SHAPE1_UPPER,
            "i11": self.i11,
            "i21": self.i21,
            "Ncases": self.Ncases
            }

        return dd


class StanSamplingDataset():
    def __init__(self, stan_variables, copula_name, names=["y", "z"]):
        # Set stan variables
        if len(stan_variables) != 2:
            errmess = "Only bivariate case accepted."
            raise ValueError(errmess)

        # .. changing names to y and z (Stan code requirement)
        stan_variables[0].name = names[0]
        stan_variables[1].name = names[1]
        if len(set([sv.N for sv in stan_variables])) != 1:
            errmess = "Differing data size"
            raise ValueError(errmess)

        self._stan_variables = stan_variables

        # Set copula
        if copula_name not in COPULA_NAMES_STAN:
            errmess = f"Copula {copula_name} is not supported."
            raise ValueError(errmess)

        self._copula_name = copula_name
        self._copula = factory(copula_name)

        # Set indexes
        self.set_indexes()
        self.set_initial_parameters()

    @property
    def copula_name(self):
        return self._copula_name

    @property
    def copula_code(self):
        return COPULA_NAMES[self._copula_name]

    @property
    def stan_variables(self):
        return self._stan_variables

    @property
    def initial_parameters(self):
        if len(self._initial_parameters) == 0:
            errmess = "Initial parameters have not been set."
            raise ValueError(errmess)

        return self._initial_parameters

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

        self.i11 = np.where(yobs & zobs)[0] + 1
        self.i21 = np.where(ycens & zobs)[0] + 1
        self.i31 = np.where(ymiss & zobs)[0] + 1

        self.i12 = np.where(yobs & zcens)[0] + 1
        self.i22 = np.where(ycens & zcens)[0] + 1
        self.i32 = np.where(ymiss & zcens)[0] + 1

        self.i13 = np.where(yobs & zmiss)[0] + 1
        self.i23 = np.where(ycens & zmiss)[0] + 1
        self.i33 = np.where(ymiss & zmiss)[0] + 1
        if len(self.i33) > 0:
            errmess = "Expected at least one variable to be valid."
            raise ValueError(errmess)

        self.Ncases = np.array([
                        [len(self.i11), len(self.i12), len(self.i13)],
                        [len(self.i21), len(self.i22), len(self.i23)],
                        [len(self.i31), len(self.i32), len(self.i33)]
                        ])
        if N != np.sum(self.Ncases):
            errmess = f"Expected total of Ncases to be {N},"\
                      + f" got {np.sum(self.Ncases)}."

    def set_initial_parameters(self):
        copula = self._copula
        inits = []
        ninits = min([vs.ninits for vs in self._stan_variables])
        for i in range(ninits):
            init = {}
            cdfs = []
            for ivs, vs in enumerate(self._stan_variables):
                n = vs.name
                for pn, value in vs.initial_parameters[i].items():
                    ppn = f"{n}{pn}"
                    init[ppn] = value

                cdfs.append(vs.initial_cdfs[ivs])

            rho = np.random.normal(loc=RHO_PRIOR[0], scale=0.2)
            rho_min = self._copula.rho_min+0.05
            rho_max = self._copula.rho_max-0.05
            rho = max(min(rho, rho_max), rho_min)

            # Check likelihood and reduce correlation if needed
            cdfs = np.column_stack(cdfs)
            notnan = ~np.any(np.isnan(cdfs), axis=1)
            cdfs = cdfs[notnan]
            niter = 0
            copula.rho = rho
            while True and niter < 10:
                copula_pdf = copula.pdf(cdfs)
                notok = np.any(np.isnan(copula_pdf))
                if notok:
                    copula.rho = copula.rho*0.5
                    valid = False
                    niter += 1
                else:
                    valid = True
                    break

            if valid:
                init["rho"] = copula.rho
            else:
                errmess = f"Cannot find initial copula parameter (rho={rho})."
                raise ValueError(errmess)

            inits.append(init)

        self._initial_parameters = inits

    def to_dict(self):
        dd = self.stan_variables[0].to_dict()
        dd.update(self.stan_variables[1].to_dict())
        dd.update({
            "copula": self.copula_code,
            "rho_lower": self._copula.rho_min,
            "rho_upper": self._copula.rho_max,
            "rho_prior": RHO_PRIOR,
            "Ncases": self.Ncases,
            "i11": self.i11,
            "i21": self.i21,
            "i31": self.i31,
            "i12": self.i12,
            "i22": self.i22,
            "i32": self.i32,
            "i13": self.i13,
            "i23": self.i23
            })
        return dd


class StanDiscreteVariable():
    def __init__(self, data, discrete_name,
                 ninits=NCHAINS_DEFAULT):
        self.name = "k"
        self._N = 0
        self._data = None
        self._discrete_code = None
        self._discrete_name = None
        self._initial_parameters = []
        self.ninits = ninits

        # Set if the 2 inputs are set
        if data is not None and discrete_name is not None:
            self.set_data(data, discrete_name)
            data_set = True

        if data_set:
            self.set_initial_parameters()

    @property
    def N(self):
        return self._N

    @property
    def initial_parameters(self):
        if len(self._initial_parameters) == 0:
            errmess = f"Variable {self.name}: "\
                      + "Initial parameters have not been set."
            raise ValueError(errmess)

        return self._initial_parameters

    @property
    def discrete_name(self):
        return self._discrete_name

    @property
    def discrete_code(self):
        return DISCRETE_NAMES[self._discrete_name]

    @property
    def data(self):
        if self._data is None:
            errmess = "Data has not been set."
            raise ValueError(errmess)
        return self._data

    def set_data(self, data, discrete_name):
        if discrete_name not in DISCRETE_NAMES:
            errmess = f"Cannot find discrete {discrete_name}."
            raise ValueError(errmess)

        self._discrete_name = discrete_name

        data = np.array(data).astype(np.int64)
        if data.ndim != 1:
            errmess = "Expected data as 1d array."
            raise ValueError(errmess)

        if np.any(data < 0):
            errmess = "Need all data to be >=0."
            raise ValueError(errmess)

        nmax = 1 if discrete_name == "Bernoulli" else NEVENT_UPPER
        if np.any(data > nmax):
            errmess = f"Need all data to be <={nmax}."
            raise ValueError(errmess)

        self._data = data
        self._N = len(data)

    def set_initial_parameters(self):
        inits = []
        for i in range(self.ninits):
            locn = self.data.mean()*max(0.05, np.random.normal(scale=0.2))
            inits.append({
                "locn": locn,
                "phi": 1
                })
        self._initial_parameters = inits

    def to_dict(self):
        """ Export stan data to be used by stan program """
        vn = self.name
        isbern = self.discrete_name == "Bernoulli"
        dd = {
            f"{vn}disc": self.discrete_code,
            "N": self.N,
            vn: self.data,
            "locn_upper": 1 if isbern else LOCN_UPPER,
            "phi_lower": PHI_LOWER,
            "phi_upper": PHI_UPPER,
            "nevent_upper": 1 if isbern else NEVENT_UPPER,
            f"{vn}locn_prior": [0.5, 3] if isbern else DISCRETE_LOCN_PRIOR,
            f"{vn}phi_prior": DISCRETE_PHI_PRIOR
            }

        return dd
