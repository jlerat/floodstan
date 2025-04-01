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
from floodstan.marginals import _prepare_censored_data

MARGINAL_CODES = {code: name for name, code in MARGINAL_NAMES.items()}
COPULA_CODES = {code: name for name, code in COPULA_NAMES.items()}

# Subset of copula currently supported in the stan model
COPULA_NAMES_STAN = ["Gaussian", "Clayton", "Gumbel"]

# Prior on copula parameter
RHO_PRIOR = [0.7, 1.]

# Prior on discrete parameters
DISCRETE_LOCN_PRIOR = [1, 10]
DISCRETE_PHI_PRIOR = [1, 10]

CENSOR_DEFAULT = -1e10

STAN_VARIABLE_INITIAL_CDF_MIN = 1e-8

MAX_INIT_PARAM_SEARCH = 50

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
    marginal.locn = locn
    marginal.logscale = logscale
    if marginal.has_shape:
        marginal.shape1 = shape1

    cdf = marginal.cdf(data)
    cdf[data < censor] = np.nan
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


def bootstrap(marginal, data, fit_method="params_guess",
              nboot=10000, eta=0):
    expected = ["fit_lh_moments", "params_guess"]
    if fit_method not in expected:
        errmess = f"Expected fit_method in [{'/'.join(expected)}]."
        raise ValueError(errmess)

    data = np.array(data)
    nval = len(data)

    # Parameter estimation method
    fun = getattr(marginal, fit_method)
    kw = {"eta": eta} if fit_method == "fit_lh_moments" else {}

    # Prepare data
    boots = pd.DataFrame(np.nan, index=np.arange(nboot),
                         columns=PARAMETERS)
    # Run bootstrap
    for i in range(nboot):
        resampled = np.random.choice(data, nval, replace=True)
        try:
            fun(resampled, **kw)
        except Exception:
            continue

        boots.loc[i, :] = marginal.params

    return boots


def importance_sampling(marginal, data, params, censor=-np.inf,
                        nsamples=10000):
    """ See
    Smith, A. F. M., & Gelfand, A. E. (1992).
    Bayesian Statistics without Tears: A Sampling-Resampling Perspective.
    The American Statistician, 46(2), 84–88. https://doi.org/10.2307/2684170
    """
    data, dcens, ncens = _prepare_censored_data(data, censor)
    params = np.array(params)

    # Maximum number of iterations
    niter_max = 3
    # Minimum ratio between neff and nsamples to stop iterating
    neff_factor = 0.5
    # Minimum neff tolerated during iteration
    neff_min = 5

    for niter in range(niter_max):
        # Compute log posteriors
        logposts = np.zeros(len(params))
        for i, param in enumerate(params):
            lp = -marginal.neglogpost(param, dcens, censor, ncens)
            logposts[i] = -1e100 if np.isnan(lp) or not np.isfinite(lp) else lp

        # Compute rescaled pdf (normalized by lp_max to avoid underflow)
        lp_max = np.nanmax(logposts)
        weights = np.exp(logposts - lp_max)
        weights /= weights.sum()
        neff = 1. / (weights**2).sum()

        if neff < neff_min:
            errmess = f"Effective sample size < {neff_min} ({neff})."
            raise ValueError(errmess)

        k = np.random.choice(np.arange(len(weights)), size=nsamples, p=weights)
        samples = params[k]
        logposts = logposts[k]

        if niter == 0:
            mean = samples.mean(axis=0)
            cov = np.cov(samples.T)
            params = np.random.multivariate_normal(mean=mean, cov=cov,
                                                   size=nsamples)

        if neff > neff_factor * nsamples:
            break

    samples = pd.DataFrame(samples, columns=PARAMETERS)

    return samples, logposts, neff


class StanSamplingVariable():
    def __init__(self,
                 marginal,
                 data=None,
                 censor=CENSOR_DEFAULT,
                 name="y",
                 ninits=NCHAINS_DEFAULT):
        self.name = str(name)
        if len(self.name) != 1:
            errmess = "Expected one character for name."
            raise ValueError(errmess)

        self._N = 0
        self._data = None
        self._censor = float(censor)
        self._guess_parameters = []
        self._initial_parameters = []
        self._is_miss = None
        self._is_obs = None
        self._is_cens = None

        self.marginal = marginal

        # Initial parameters
        self.ninits = ninits
        self._initial_parameters = []
        self._initial_cdfs = []

        data_set = False
        if data is not None and censor is not None:
            self.set_data(data, censor)
            data_set = True

        if data_set:
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
        # .. 3x3 due to stan code requirement.
        #    Only first 2 top left cells used
        self.Ncases = np.zeros((3, 3), dtype=int)
        self.Ncases[:2, 0] = [len(self.i11), len(self.i21)]

        # We set the censor close to data min to avoid potential
        # problems with computing log cdf for the censor
        censor = max(np.float64(censor), dok.min() - 1e-10)
        self._censor = censor

    def set_guess_parameters(self):
        censor = self.censor
        data, dcens, ncens = _prepare_censored_data(self.data, censor)
        dist = self.marginal
        dist.params_guess(dcens)
        self._guess_parameters = {
                "locn": dist.locn,
                "logscale": dist.logscale,
                "shape1": dist.shape1
                }

    def set_initial_parameters(self):
        censor = self.censor
        data, dcens, ncens = _prepare_censored_data(self.data, censor)
        dist = self.marginal

        # Default parameters
        dist.params_guess(dcens)
        params0 = dist.params

        # Create a parameter from bootstrap guess
        # parameters for each chain
        ninits = self.ninits
        niter = 0
        inits, cdfs = [], []
        nval = len(data)
        k = np.where(~np.isnan(data))[0]

        while len(inits) < ninits \
                and niter < MAX_INIT_PARAM_SEARCH:
            niter += 1
            kk = np.random.choice(k, nval, replace=True)
            dboot = data[kk]

            # Errors can occur here as params guess
            # is not necessarily conform with prior
            try:
                dist.params_guess(dboot)
            except ValueError:
                continue

            locn, logscale, shape1 = dist.params
            pp, cdf = are_marginal_params_valid(dist, locn, logscale,
                                                shape1, data, censor)
            if pp is not None:
                inits.append(pp)
                cdfs.append(cdf)

        # Fill up inits with params0 with small random noise
        if len(inits) < ninits:
            locn0, logscale0, shape10 = params0
            for i in range(ninits - len(inits)):
                eps = np.random.uniform(-1, 1, size=3) * 1e-5
                locn = locn0 + eps[0]
                logscale = logscale0 + eps[1]
                shape1 = shape10 + eps[2]

                pp, cdf = are_marginal_params_valid(dist, locn, logscale,
                                                    shape1, data, censor)
                if pp is not None:
                    inits.append(pp)
                    cdfs.append(cdf)

        if len(inits) == 0:
            errmess = "Cannot find initial parameter "\
                      + f"for variable {self.name}."
            raise ValueError(errmess)

        if len(inits) < ninits:
            nok = len(inits)
            for i in range(ninits - nok):
                k = np.random.choice(np.arange(nok))
                inits.append(inits[k])

        self._initial_parameters = inits
        self._initial_cdfs = cdfs

    def set_priors(self):
        # Special set for LogPearson3 due to
        # fitting difficulties otherwise
        # use prior from marginal
        if self.marginal.name == "LogPearson3":
            start = self.guess_parameters
            self.marginal.locn_prior.loc = start["locn"]
            self.marginal.logscale_prior.loc = start["logscale"]
            self.marginal.shape1_prior.loc = start["shape1"]

    def to_dict(self):
        """ Export stan data to be used by stan program """
        vn = self.name
        dist = self.marginal
        dd = {
            f"{vn}marginal": self.marginal_code,
            "N": self.N,
            vn: self.data,
            f"{vn}censor": self.censor,
            f"{vn}locn_prior": dist.locn_prior.to_list(),
            f"{vn}logscale_prior": dist.logscale_prior.to_list(),
            f"{vn}shape1_prior": dist.shape1_prior.to_list(),
            "locn_lower": dist.locn_prior.lower,
            "locn_upper": dist.locn_prior.upper,
            "logscale_lower": dist.logscale_prior.lower,
            "logscale_upper": dist.logscale_prior.upper,
            "shape1_lower": dist.shape1_prior.lower,
            "shape1_upper": dist.shape1_prior.upper,
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
            for ivs, vs in enumerate(self._stan_variables):
                n = vs.name
                for pn, value in vs.initial_parameters[i].items():
                    ppn = f"{n}{pn}"
                    init[ppn] = value

                cdfs.append(vs.initial_cdfs[ivs])

            cdfs = np.column_stack(cdfs)
            notnan = ~np.any(np.isnan(cdfs), axis=1)
            cdfs = cdfs[notnan]

            niter = 0
            rho = np.nan
            nval = len(cdfs)
            k = np.arange(nval)

            while True and niter < MAX_INIT_PARAM_SEARCH:
                niter += 1
                kk = np.random.choice(k, nval, replace=True)
                rho = kendalltau(cdfs[kk, 0], cdfs[kk, 1]).statistic
                rho = min(copula.rho_max, max(copula.rho_min, rho))

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
