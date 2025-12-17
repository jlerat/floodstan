import sys
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from floodstan import NCHAINS_DEFAULT

from floodstan.copulas import COPULA_NAMES
from floodstan.marginals import MARGINAL_NAMES
from floodstan.marginals import PARAMETERS
from floodstan.copulas import factory
from floodstan.data_processing import univariate2censored
from floodstan.data_processing import univariate2cases

MARGINAL_CODES = {code: name for name, code in MARGINAL_NAMES.items()}
COPULA_CODES = {code: name for name, code in COPULA_NAMES.items()}

# Subset of copula currently supported in the stan model
COPULA_NAMES_STAN = ["Gaussian", "Clayton", "Gumbel"]

CENSOR_DEFAULT = -1e10

STAN_VARIABLE_INITIAL_CDF_MIN = 1e-8

# Maximum number of parameter sampled for initialisation
NPARAMS_INITS_MAX = 1000

# Special stan sampling arguments
STAN_SAMPLE_ARGS = {
        "LogPearson3": {"adapt_delta": 0.999},
        "GeneralizedLogistic": {"adapt_delta": 0.999},
        "GeneralizedPareto": {"adapt_delta": 0.999},
        "GEV": {"adapt_delta": 0.99},
        }

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


def _check_prior(values):
    values = np.array(values).astype(float)
    if values.shape != (2, ):
        errmess = f"Expected an array of shape (2, ), got {values.shape}."
        raise ValueError(errmess)
    return values


def are_marginal_params_valid(marginal, locn, logscale, shape1, data, censor):
    try:
        marginal.locn = locn
        marginal.logscale = logscale
        if marginal.has_shape:
            marginal.shape1 = shape1
    except Exception:
        return None, None

    # Check prior
    for pn in PARAMETERS:
        prior = getattr(marginal, f"{pn}_prior")
        v = marginal[pn]
        if v < prior.lower or v > prior.upper:
            return None, None

    # Check likelihood
    cdf = marginal.cdf(data)
    cdf[data < censor] = np.nan
    cdf[np.isnan(data)] = np.nan
    cdf_min = np.nanmin(cdf)
    cdf_max = np.nanmax(cdf)
    cdf_censor = marginal.cdf(censor)

    cmin = STAN_VARIABLE_INITIAL_CDF_MIN
    isok = (cdf_min >= cmin) & (cdf_max <= 1 - cmin)
    isok &= (cdf_censor >= cmin) & (cdf_censor <= 1 - cmin)

    if isok:
        dd = {"locn": marginal.locn,
              "logscale": marginal.logscale,
              "shape1": marginal.shape1
              }
        return dd, cdf

    return None, None


class StanSamplingVariable():
    def __init__(self,
                 marginal,
                 data,
                 censor=CENSOR_DEFAULT,
                 name="y",
                 ninits=NCHAINS_DEFAULT,
                 nparams_sampled=None):
        self._name = str(name)
        if len(self._name) != 1:
            errmess = "Expected one character in name."
            raise ValueError(errmess)

        self._N = 0
        self._data = None
        self._censor = float(censor)
        self._guess_parameters = []
        self._initial_parameters = []
        self._is_miss = None
        self._is_obs = None
        self._is_cens = None

        self.marginal = marginal.clone()

        # Initial parameters
        self.ninits = ninits
        self._initial_parameters = []
        self._initial_cdfs = []

        # Parameters sampled for initial and potentially prior
        nparams_sampled = NPARAMS_INITS_MAX if nparams_sampled is None\
            else nparams_sampled

        # Importance sampling
        self.nparams_sampled = nparams_sampled
        self._sampled_parameters = []
        self._sampled_parameters_valid = []

        self.set_data(data, censor)
        self.set_guess_parameters()
        self.set_sampled_parameters()
        self.set_initial_parameters()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        name = str(name)
        if len(name) != 1:
            errmess = "Expected one character in name."
            raise ValueError(errmess)
        self._name = name

        guess = self._guess_parameters
        if len(guess) != 0:
            self._guess_parameters = \
                {f"{name}{k[1:]}": v for k, v in guess.items()}

        inits = self._initial_parameters
        if len(inits) != 0:
            for i in range(len(inits)):
                p = inits[i]
                inits[i] = {f"{name}{k[1:]}": v for k, v in p.items()}

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
    def sampled_parameters(self):
        if len(self._sampled_parameters) == 0:
            errmess = "Importance parameters have not been set."
            raise ValueError(errmess)
        return self._sampled_parameters

    @property
    def sampled_parameters_valid(self):
        if len(self._sampled_parameters_valid) == 0:
            errmess = "Importance parameters have not been set."
            raise ValueError(errmess)
        return self._sampled_parameters_valid

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
    def marginal_code(self):
        return MARGINAL_NAMES[self.marginal.name]

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
    def stan_sample_args(self):
        marginal_name = self.marginal.name
        if marginal_name in STAN_SAMPLE_ARGS:
            return STAN_SAMPLE_ARGS[marginal_name]
        else:
            return {}

    def set_data(self, data, censor):
        icases, data, censor = univariate2cases(data, censor)
        N = len(data)
        self._N = N
        self._data = data
        self._censor = censor
        self._is_obs = icases.i11
        self._is_cens = icases.i21
        self._is_miss = icases.i31
        self.i11 = np.where(icases.i11)[0] + 1
        self.i21 = np.where(icases.i21)[0] + 1

        # .. 3x3 due to stan code requirement.
        #    Only first 2 top left cells used
        self.Ncases = np.zeros((3, 3), dtype=int)

        N11 = len(self.i11)
        N21 = len(self.i21)
        self.Ncases[:, 0] = [N11, N21, N - N11 - N21]

    def set_guess_parameters(self):
        censor = self.censor
        _, dcens, _, _ = univariate2censored(self.data, censor)
        dist = self.marginal
        dist.params_guess(dcens)
        name = self.name
        self._guess_parameters = {
                f"{name}locn": dist.locn,
                f"{name}logscale": dist.logscale,
                f"{name}shape1": dist.shape1
                }

    def set_sampled_parameters(self):
        nsamples = self.nparams_sampled

        # Get guess parameters
        n = self.name
        p0 = np.array([self.guess_parameters[f"{n}{pn}"]
                       for pn in PARAMETERS])

        # Perturb guess parameters
        eps = np.random.normal(scale=2e-1, size=(nsamples, 3))
        params = pd.DataFrame(p0[None, :] + eps, columns=PARAMETERS)
        params.loc[:, "locn"] = p0[0] * (1 + eps[:, 0])

        # Sample closer to 0 to avoid problems with suppoer
        params.loc[:, "shape1"] = \
            p0[2] * np.random.uniform(0, 1.0, size=nsamples)

        self._sampled_parameters = params
        self._sampled_parameters_valid = -np.ones(len(params))

    def set_initial_parameters(self):
        ninits = self.ninits
        niter = 0
        inits, cdfs = [], []
        marginal = self.marginal
        data = self.data
        censor = self.censor
        params = self.sampled_parameters

        while len(inits) < ninits and niter < len(params):
            locn, logscale, shape1 = params.iloc[niter]
            pp, cdf = are_marginal_params_valid(marginal, locn, logscale,
                                                shape1, data, censor)
            if pp is None:
                self._sampled_parameters_valid[niter] = 0
            else:
                n = self.name
                pp = {f"{n}{pn}": v for pn, v in pp.items()}
                inits.append(pp)
                cdfs.append(cdf)
                self._sampled_parameters_valid[niter] = 1

            niter += 1

        if len(inits) < ninits:
            errmess = "Cannot find initial parameter "\
                      + f"for variable {self.name}."
            raise ValueError(errmess)

        self._initial_parameters = inits
        self._initial_cdfs = cdfs

    def to_dict(self):
        """ Export stan data to be used by stan program """
        vn = self.name
        dist = self.marginal
        dd = {
            f"{vn}marginal": self.marginal_code,
            "N": self.N,
            vn: self.data.tolist(),
            f"{vn}censor": float(self.censor),
            f"{vn}locn_prior": dist.locn_prior.to_list(),
            f"{vn}logscale_prior": dist.logscale_prior.to_list(),
            f"{vn}shape1_prior": dist.shape1_prior.to_list(),
            "locn_lower": float(dist.locn_prior.lower),
            "locn_upper": float(dist.locn_prior.upper),
            "logscale_lower": float(dist.logscale_prior.lower),
            "logscale_upper": float(dist.logscale_prior.upper),
            "shape1_lower": float(dist.shape1_prior.lower),
            "shape1_upper": float(dist.shape1_prior.upper),
            "i11": self.i11.tolist(),
            "i21": self.i21.tolist(),
            "Ncases": self.Ncases.tolist()
            }

        return dd


class StanSamplingDataset():
    def __init__(self, stan_variables, copula_name, names=["y", "z"]):
        # Set stan variables
        if len(stan_variables) != 2:
            errmess = "Only bivariate case accepted."
            raise ValueError(errmess)

        ndata = set(len(v.data) for v in stan_variables)
        if len(ndata) > 1:
            errmess = "Expected all stan variables to have "\
                      + "the same number of data samples."
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

    @property
    def stan_sample_args(self):
        say = self.stan_variables[0].stan_sample_args
        saz = self.stan_variables[1].stan_sample_args
        if say != saz:
            errmess = "Both variables should have the same stan_sample_args."
            raise ValueError(errmess)
        return say

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

        # Finds number of initial parameters
        # (minimum of number of initial params for each variable)
        ninits = min([vs.ninits for vs in self._stan_variables])

        for i in range(ninits):
            init = {}
            cdfs = []
            data = []
            for ivs, vs in enumerate(self._stan_variables):
                data.append(vs.data)
                for pn, value in vs.initial_parameters[i].items():
                    init[pn] = value

                cdfs.append(vs.initial_cdfs[ivs])

            cdfs = np.column_stack(cdfs)
            notnan = ~np.any(np.isnan(cdfs), axis=1)
            cdfs = cdfs[notnan]

            iok = ~np.isnan(data[0]) & ~np.isnan(data[1])
            rho0 = kendalltau(data[0][iok], data[1][iok]).statistic
            rhos = np.random.normal(loc=rho0, scale=0.1,
                                    size=NPARAMS_INITS_MAX)
            rhos = rhos.clip(copula.rho_lower + 0.02,
                             copula.rho_upper - 0.02)
            niter = 0
            while True and niter < NPARAMS_INITS_MAX:
                rho = rhos[niter]
                niter += 1

                # Check likelihood
                copula.rho = rho
                copula_pdfs = copula.pdf(cdfs)
                isok = np.all(~np.isnan(copula_pdfs))
                if isok:
                    break

            if np.isnan(rho):
                errmess = "Cannot initialise parameters for dataset."
                raise ValueError(errmess)

            init["rho"] = rho
            inits.append(init)

        self._initial_parameters = inits

    def to_dict(self):
        dd = self.stan_variables[0].to_dict()
        dd.update(self.stan_variables[1].to_dict())

        # Careful with upper and lower bounds
        for pn in PARAMETERS:
            p0, p1 = np.inf, -np.inf
            for v in self.stan_variables:
                prior = getattr(v.marginal, f"{pn}_prior")
                p0 = min(p0, prior.lower)
                p1 = max(p1, prior.upper)

            dd[f"{pn}_lower"] = float(p0)
            dd[f"{pn}_upper"] = float(p1)

        rho_prior = self._copula.rho_prior

        dd.update({
            "copula": int(self.copula_code),
            "rho_lower": float(rho_prior.lower),
            "rho_upper": float(rho_prior.upper),
            "rho_prior": rho_prior.to_list(),
            "Ncases": self.Ncases.tolist(),
            "i11": self.i11.tolist(),
            "i21": self.i21.tolist(),
            "i31": self.i31.tolist(),
            "i12": self.i12.tolist(),
            "i22": self.i22.tolist(),
            "i32": self.i32.tolist(),
            "i13": self.i13.tolist(),
            "i23": self.i23.tolist()
            })
        return dd
