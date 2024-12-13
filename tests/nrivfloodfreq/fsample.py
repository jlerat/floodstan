import re, math
from pathlib import Path
import shutil
import logging

from datetime import datetime
import time

import numpy as np
import pandas as pd

from scipy.stats import skew

from cmdstanpy import CmdStanModel

from tqdm import tqdm

from hydrodiy.stat import sutils
from nrivfloodfreq.fdist import _check_1d_data

# Path to stan models
FSTANFILES = Path(__file__).resolve().parent / "stan_models"
FSTANEXE = FSTANFILES.parent / "stan_executables"
FSTANEXE.mkdir(exist_ok=True)

# Setup logger with a write function to use contextlib
LOGGER = logging.getLogger("cmdstanpy")

# List of stan model
STAN_MODEL_NAMES = ["censoring_noerrobs_nocovariate", \
                    "censoring_errobs_nocovariate", \
                    "censoring_errobs_covariate", \
                    "censoring_noerrobs_covariate", \
                    "censoring_noerrobs_covariatecensored", \
                    "censoring_errobs_covariatecensored"]

STAN_SEED = 5446

# Priors for streamflow error model
MU_ERR = 1
SIG_ERR = 0.2
MIN_ERR = 0.2 # Min error of q/5
MAX_ERR = 5   # Max error of q*5

# Prior for copula correlation
MSHIFTED_PRIOR = np.array([3, 6])
MSHIFTED_MIN = 0.01
MSHIFTED_MAX = 19

# GEV prior for kappa parameter
KAPPA_LOWER = -3
KAPPA_UPPER = 3
KAPPA_PRIOR = np.array([0, 4]) # mean and stdvev of normal prior

TAU_PRIOR_SIG_FACTOR = 2 # multiplier for mean of normal prior to get stdev
LOGALPHA_PRIOR_SIG = 2 # stdev of normal prior

# Prior for LogPearson3
G_LOWER = -1.9
G_UPPER = 1.9
G_PRIOR = np.array([0, 4])

M_PRIOR_SIG_FACTOR = 2 # multiplier for mean of normal prior to get stdev
LOGS_PRIOR_SIG = 2 # stdev of normal prior


def set_logger(flog, level="INFO", \
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s"):
    """ Get a logger object that can handle contextual info

    Parameters
    -----------
    flog : str
        Path to log file. If none, no log file is used.
    level : str
        Logging level.
    fmt : str
        Log format

    Returns
    -----------
    logger : logging.Logger
        Logger instance
    """
    # Set logging level
    if not level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"{level} not a valid level")

    LOGGER.setLevel(getattr(logging, level))

    # Set logging format
    ft = logging.Formatter(fmt)

    # log to file
    flog = Path(flog)
    fh = logging.FileHandler(flog)
    fh.setFormatter(ft)
    LOGGER.addHandler(fh)

    LOGGER.info("Process started")



def compute_samples_quantiles(dist, samples, \
            return_periods=[2, 5, 10, 20, 30, 50, 100, 200, 500, 1000], \
            data_length_adjustment=1.):
    """ Compute flood quantiles for selected return periods

    Parameters
    ----------
    dist : FloodFreqDistribution
        Flood frequency distribution.
    samples : pandas.DataFrame
        Parameter samples
    return periods : list
        List of return period to compute flood quantiles for.
    data_length_adjustment : float

    """
    errmsg = "Expected data length adjustment in ]0, 1], got"+\
                f"{data_length_adjustment}"
    assert data_length_adjustment<=1 and data_length_adjustment>0, errmsg

    # Check inputs
    return_periods = np.array(return_periods)
    nsamples, nquantiles = len(samples), len(return_periods)
    samples_quantiles = pd.DataFrame(np.nan, index=samples.index, \
                                columns=return_periods)
    p = 1-1/return_periods*data_length_adjustment
    return pd.DataFrame(dist.ppfvect(p, samples), \
                                index=samples.index, \
                                columns=return_periods)


def compute_expected_params_quantiles(dist, samples, weights=None, \
                            return_periods=[2, 5, 10, 20, 30, 50, 100, 200, 500, 1000], \
                            back_transform_log_params=True):
    """ Flood quantile corresponding to expected parameters.
        See ARR, Book 3, section 2.6.3.7 page 43.
    """
    # Check inputs
    assert isinstance(samples, pd.DataFrame)
    return_periods = np.array(return_periods)
    # .. same weights if weigh vector is none
    nsamples = len(samples)
    weights = np.ones(nsamples)/nsamples if weights is None else weights
    weights /= weights.sum()

    # Convert log transformed parameters to original space
    if back_transform_log_params:
        expon = dist.samples2expon(samples)
        expon_means = (expon*weights[:, None]).sum(axis=0)
        expon_means = pd.DataFrame(np.array(expon_means)[None, :], \
                            columns=expon.columns)
        means = dist.expon2samples(expon_means).loc[0]
    else:
        means = (samples*weights[:, None]).sum(axis=0)

    # Compute expected parameter quantile
    dist.set_dict_params(means)
    exp_param_qt = pd.Series(dist.ppf(1-1/return_periods), index=return_periods)

    return exp_param_qt, means


def compute_quantiles_stats(samples_quantiles, weights=None):
    """ Probabilistic estimate of quantiles using weighted samples.
        See ARR, Book 3, section 2.6.3.7 page 43.
    """
    # Check inputs
    assert isinstance(samples_quantiles, pd.DataFrame)
    return_periods = samples_quantiles.columns
    # .. same weights if weigh vector is none
    nsamples = len(samples_quantiles)
    weights = np.ones(nsamples)/nsamples if weights is None else weights
    weights /= weights.sum()

    # Create result dataframe
    cols =["0.05%", "0.5%", "2.5%", "5%", "10%", "20%", "50%", "80%", \
                "90%", "95%", "97.5%", "99.5%", "99.95%", "mean"]
    qq = [float(re.sub("%", "", x))/100 for x in cols[:-1]]
    quantiles_stats = pd.DataFrame(np.nan, index=return_periods, columns=cols)

    # Compute quantiles stats
    for i, retper in enumerate(return_periods):
        # Quantiles
        k = np.argsort(samples_quantiles.values[:, i])
        cdf = np.cumsum(weights[k])
        st = np.interp(qq, cdf, samples_quantiles.values[k, i])
        # Mean
        m = np.sum(samples_quantiles.values[:, i]*weights)
        # Store
        quantiles_stats.iloc[i, :] = np.append(st, [m])

    return quantiles_stats


def bootstrap_lh_moments(dist, streamflow, nsamples, eta=0):
    """ Run a parametric bootstram sampling for LH moment fit.

    Parameters
    ----------
    dist : FloodFreqDistribution
        Flood frequency distribution.
    streamflow : numpy.ndarray
        Streamflow observations.
    nsamples : int
        Number of samples to generate.
    eta : int
        LH moments shift.

    Returns
    -------
    samples : np.ndarray
        Parameter samples generated.
        Array [nsamples, nparams].
    y : np.ndarray
        Data generated.
        Array [ndata, nsamples].
    """
    # Initialise / check data
    _check_1d_data(streamflow)
    nobs = len(streamflow)
    samples = []

    # Initial fit
    params = dist.fit_lh_moments(streamflow, eta=eta)
    dist.set_dict_params(params.loc[0])

    # Sample random obs
    y = dist.rvs((nobs, nsamples))

    # Fit data
    samples = dist.fit_lh_moments(y, eta=eta)

    return samples, y


def get_stan_model(dist, stan_model_name, force_compile=False, wait_seconds=300):
    """ Compile a stan model or open it if compiled. """
    assert stan_model_name in STAN_MODEL_NAMES
    distribution = dist.name

    if re.search("_covariate", stan_model_name) and distribution != "GEV":
        errmsg = f"Covariate model only available for GEV, got dist.name={dist.name}."
        raise ValueError(errmsg)

    stan_file = FSTANFILES / f"{distribution.lower()}_{stan_model_name}.stan"
    if not stan_file.exists():
        errmsg = f"No stan file {stan_file}."
        raise FileNotFoundError(errmsg)

    exe_file = FSTANEXE / stan_file.stem
    wait_file = FSTANEXE / f"{stan_file.stem}_compiling_flag.txt"

    if not exe_file.exists() or force_compile:
        # Delete files if force compile
        if force_compile:
            if exe_file.exists():
                exe_file.unlink()
            if wait_file.exists():
                wait_file.unlink()

        LOGGER.info("Stan model executable does not exist or forced recompilation.")
        if wait_file.exists():
            with wait_file.open("r") as wf:
                txt = wf.read()
                if not re.search("ERROR", txt):
                    LOGGER.info(".. a waiting file exists, "+\
                            f"so we wait {wait_seconds}s for compilation to finish")
                    time.sleep(wait_seconds)
                else:
                    LOGGER.error(".. waiting file contains error. "+\
                                    "Restart compilation.")

        LOGGER.info("Stan model compilation started.")
        with wait_file.open("w") as wf:
            start = datetime.now()
            wf.write(f"Compiling of model [{distribution}] started at {start}\n")

            try:
                # Copy files to exe folders
                lf = [stan_file]+list(FSTANFILES.glob("*.stanfunctions"))
                for f in lf:
                    shutil.copyfile(f, FSTANEXE / f.name)

                # Compile model
                stan_file_copy = FSTANEXE / stan_file.name
                model = CmdStanModel(stan_file=stan_file_copy)
            except Exception as err:
                LOGGER.error(f"Stan model compilation aborted:\n\n{err}")
                wf.write(f"ERROR:\n\n{err}")
                raise err

            end = datetime.now()
            delta = end-start
            message = f"Compiling of model [{distribution}] completed at {end} ({delta.seconds} sec)"
            wf.write(f"{message}\n")
            LOGGER.info(message)

    else:
        LOGGER.info("Stan model executable exists. Loading model.")
        # Compilation avaiable, clean the wait file
        if wait_file.exists():
            time.sleep(2)
            wait_file.unlink()

        # load the model
        model = CmdStanModel(exe_file=exe_file)

    return model


def _adjust_gev_parameters(y, tau, logalpha, kappa, tol_y_factor=0.05, \
                                tol_kappa=0.05):
    alpha = math.exp(logalpha)

    # Compute y tolerance from mean(y) and factor
    tol_y = np.nanmean(y)*tol_y_factor

    # Bound of permissible range
    y0 = tau+alpha/kappa

    # Restrict kappa to permissible range
    #kappa = max(min(kappa, KAPPA_UPPER-0.05), KAPPA_LOWER+0.05)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)

    if kappa<0 and ymin<y0+tol_y:
        # support has an lower bound
        # .. adjust kappa first
        kappa = alpha/(ymin-tau-tol_y)

        # .. check kappa is within max bounds with tolerance
        kappa = max(KAPPA_LOWER+tol_kappa, \
                    min(KAPPA_UPPER-tol_kappa, kappa))

        # .. re-check permissible range
        y1 = tau+alpha/kappa
        if ymin<y1+tol_y:
            # .. adjust alpha to match range
            alpha = kappa*(ymin-tau-tol_y)

        y2 = tau+alpha/kappa
        errmsg = f"Kappa({kappa:0.2f})<0 and ymin({ymin:0.2f})<"+\
                    f"tau+alpha/kappa({y2:0.2f})."
        assert ymin>y2, errmsg


    if kappa>0 and ymax>y0-tol_y:
        # support has an upper bound
        # .. adjust kappa first
        kappa = alpha/(ymax-tau+tol_y)

        # .. check kappa is within max bounds with tolerance
        kappa = max(KAPPA_LOWER+tol_kappa, \
                    min(KAPPA_UPPER-tol_kappa, kappa))

        # .. re-check permissible range
        y1 = tau+alpha/kappa
        if ymax>y0-tol_y:
            # .. adjust alpha to match range
            alpha = kappa*(ymax-tau+tol_y)

        y2 = tau+alpha/kappa
        errmsg = f"Kappa({kappa:0.2f})<0 and ymax({ymax:0.2f})>"+\
                    f"tau+alpha/kappa({y2:0.2f})."
        assert ymax<y2, errmgs


    logalpha = math.log(alpha)
    return tau, logalpha, kappa


def prepare_stan_data(dist, streamflow, censor=None, anchor=None, \
                                        covariate=None, censorz=None):
    """ Prepare data for stan sampling """
    # Initialise / check data
    if covariate is None:
        idx = pd.notnull(streamflow)
        streamflow = np.array(streamflow[idx]).astype(np.float64)
        _check_1d_data(streamflow)

        has_covariate = False
        covariate = np.ones_like(streamflow)
    else:
        errmsg = "Expected streamflow and covariate to be of same length."
        assert len(streamflow) == len(covariate), errmsg

        if isinstance(streamflow, pd.Series):
            errmsg = "Streamflow is a pandas Series, expect covariate to be the same."
            assert isinstance(covariate, pd.Series), errmsg

            errmsg = "Expected streamflow and covariate to have same indexes."
            assert np.allclose(streamflow.index, covariate.index), errmsg

        elif isinstance(streamflow, np.ndarray):
            errmsg = "Streamflow is a numpy array, expect covariate to be the same."
            assert isinstance(covariate, np.ndarray), errmsg

        data = pd.DataFrame({"obs": streamflow, "covariate": covariate})
        data = data.loc[data.notnull().any(axis=1), :]
        if len(data)<5:
            errmsg = "Joint data set contains less than 5 points."
            raise ValueError(errmsg)

        streamflow = np.array(data.obs).astype(np.float64)

        has_covariate = True
        covariate = np.array(data.covariate).astype(np.float64)

    # When no censor, ensures that value is lower than
    # minimum to avoid gradient problem with absolute
    # values in stan
    censor = streamflow.min()-1e-3 if censor is None else np.float64(censor)
    assert censor>=1e-6

    if censorz is None:
        covariate_censored = False
        censorz = 1e-6
    else:
        covariate_censored = True
        censorz = np.float64(censorz)

    assert censorz>=1e-6

    anchor = censor if anchor is None else np.float64(anchor)
    assert anchor>=censor

    # censor/uncensored data
    ymiss = pd.isnull(streamflow)
    yobs = streamflow>=censor
    ycens = streamflow<censor

    zobs = covariate>=censorz
    zcens = covariate<censorz

    c11 = yobs & zobs
    c21 = ycens & zobs
    c31 = ymiss & zobs

    c12 = yobs & zcens
    c22 = ycens & zcens
    c32 = ymiss & zcens

    Ncases = np.array([[c11.sum(), c12.sum()], \
                [c21.sum(), c22.sum()], \
                [c31.sum(), c32.sum()]])

    imiss = np.where(ymiss)[0]+1
    Nmiss = len(imiss)
    icens = np.where(ycens)[0]+1
    Ncens = len(icens)
    iobs = np.where(yobs)[0]+1
    Nobs = len(iobs)

    stan_data = {
        "censor": censor, \
        "censorz": censorz, \
        # Default anchor=censor (all uncensored flow are in error)
        # See Kuczera, 1996.
        "anchor": anchor, \
        # Stdev of error as per Kuczera, 1996,
        "mu_err": MU_ERR, \
        "sig_err": SIG_ERR, \
        "min_err": MIN_ERR, \
        "max_err": MAX_ERR, \
        "mshifted_prior": MSHIFTED_PRIOR, \
        "mshifted_min": MSHIFTED_MIN, \
        "mshifted_max": MSHIFTED_MAX
    }

    if covariate_censored:
        stan_data["Ncases"] = Ncases
        stan_data["N"] = len(streamflow)

        assert Ncases[0].sum() == Nobs
        assert Ncases[1].sum() == Ncens
        assert Ncases[2].sum() == Nmiss
        assert Ncases.sum() == stan_data["N"]
    else:
        stan_data["N"] = len(streamflow)
        stan_data["Nmiss"] = Nmiss
        stan_data["Ncens"] = Ncens
        stan_data["Nobs"] = Nobs

        assert Nmiss+Ncens+Nobs == stan_data["N"]

    if has_covariate:
        y = streamflow.copy()
        # Remove censored variable to make sure stan does not use them
        y[imiss-1] = np.nan
        y[icens-1] = np.nan
        stan_data["y"] = y

        z = covariate.copy()
        stan_data["z"] = z
        stan_data["has_covariate"] = 1

        if covariate_censored:
            # Remove censored values to make sure stan does not use them
            z[c12] = np.nan
            z[c22] = np.nan
            z[c32] = np.nan
            stan_data["z"] = z

            stan_data["i11"] = np.where(c11)[0]+1
            stan_data["i21"] = np.where(c21)[0]+1
            stan_data["i31"] = np.where(c31)[0]+1
            stan_data["i12"] = np.where(c12)[0]+1
            stan_data["i22"] = np.where(c22)[0]+1
            stan_data["i32"] = np.where(c32)[0]+1
            stan_data["has_covariate_censored"] = 1

            # Check no one is missing!
            iall = np.concatenate([stan_data["i11"], \
                        stan_data["i12"], \
                        stan_data["i21"], \
                        stan_data["i22"], \
                        stan_data["i31"], \
                        stan_data["i32"]])
            iall = np.sort(iall)
            assert np.allclose(iall, np.arange(1, stan_data["N"]+1))

        else:
            stan_data["iobs"] = iobs
            stan_data["icens"] = icens
            stan_data["imiss"] = imiss
            stan_data["has_covariate_censored"] = 0

        assert len(stan_data["y"]) == stan_data["N"]
        assert len(stan_data["z"]) == stan_data["N"]

    else:
        stan_data["has_covariate"] = 0
        stan_data["has_covariate_censored"] = 0

        stan_data["y"] = streamflow[iobs-1].copy()
        stan_data["iobs"] = iobs
        stan_data["icens"] = icens
        stan_data["imiss"] = imiss

    # Add additional data depending on model
    if dist.name == "GEV":
        stan_data["kappa_prior"] = KAPPA_PRIOR
        stan_data["kappa_lower"] = KAPPA_LOWER
        stan_data["kappa_upper"] = KAPPA_UPPER

        # lh bootstrap fit to get prior location
        p, _ = bootstrap_lh_moments(dist, streamflow[iobs-1], 10000, eta=0)
        pm = p.mean()
        tau, logalpha, kappa = pm.tau, pm.logalpha, pm.kappa

        # .. adjust if parameters are not permissible
        tau, logalpha, kappa = _adjust_gev_parameters(stan_data["y"], \
                                    tau, logalpha, kappa)
        # .. store as mean prior
        stan_data["tau_prior"] = np.array([tau, TAU_PRIOR_SIG_FACTOR*tau])
        stan_data["logalpha_prior"] = np.array([logalpha, LOGALPHA_PRIOR_SIG])
        stan_data["kappa_guess"] = kappa

        if has_covariate:
            # lh bootstrap fit to get prior location
            pz, _ = bootstrap_lh_moments(dist, covariate, 10000, eta=0)
            pzm = pz.mean()
            tauz, logalphaz, kappaz = pzm.tau, pzm.logalpha, pzm.kappa

            # .. adjust if parameters are not permissible
            tauz, logalphaz, kappaz = _adjust_gev_parameters(stan_data["z"], \
                                                tauz, logalphaz, kappaz)

            # .. store as mean prior
            stan_data["tauz_prior"] = np.array([tauz, TAU_PRIOR_SIG_FACTOR*tauz])
            stan_data["logalphaz_prior"] = np.array([logalphaz, LOGALPHA_PRIOR_SIG])
            stan_data["kappaz_guess"] = kappaz

            # Correlation prior
            kt = data.iloc[iobs-1, :].corr(method="kendall").values[0, 1]
            stan_data["mshifted_guess"] = 1/(1-kt)-1

    elif dist.name == "LogPearson3":
        stan_data["g_prior"] = G_PRIOR
        stan_data["g_lower"] = G_LOWER
        stan_data["g_upper"] = G_UPPER

        # lh bootstrap fit to get prior location
        p, _ = bootstrap_lh_moments(dist, streamflow[iobs-1], 10000, eta=0)
        pm = p.mean()
        m, logs, g = pm.m, math.log(pm.s), pm.g

        stan_data["m_prior"] = np.array([m, M_PRIOR_SIG_FACTOR*m])
        stan_data["logs_prior"] = np.array([logs, LOGS_PRIOR_SIG])
        stan_data["g_guess"] = g

    return stan_data



def initialise_stan_sample(dist, stan_data):
    if dist.name == "LogNormal":
        params = dist.fit_lh_moments(stan_data["y"]).squeeze()
        inits = {\
            "m": params.m, \
            "logs": math.log(params.s)
        }

    elif dist.name == "Gumbel":
        params = dist.fit_lh_moments(stan_data["y"]).squeeze()
        inits = {\
            "tau": params.tau, \
            "logalpha": params.logalpha, \
        }

    elif dist.name == "GEV":
        tau = stan_data["tau_prior"][0]
        logalpha = stan_data["logalpha_prior"][0]
        alpha = math.exp(logalpha)
        kappa = stan_data["kappa_guess"]

        inits = {\
            "tau": tau, \
            "logalpha": logalpha, \
            "kappa": kappa
        }

        # Add covariate initialisation
        if "z" in stan_data:
            inits["tauz"] = stan_data["tauz_prior"][0]
            inits["logalphaz"] = stan_data["logalphaz_prior"][0]
            inits["kappaz"] = stan_data["kappaz_guess"]

            # Initialise correlation
            mshifted = stan_data["mshifted_guess"]
            inits["mshifted"] = max(min(mshifted, stan_data["mshifted_max"]-0.001), \
                            stan_data["mshifted_min"]+0.001)

    elif dist.name == "LogPearson3":
        m = stan_data["m_prior"][0]
        logs = stan_data["logs_prior"][0]
        s = math.exp(logs)
        g = stan_data["g_guess"]

        # Check parameter sanity
        alpha = 4/g/g
        beta = 2/g/s
        tau = m-alpha/beta
        logcensor = math.log(stan_data["censor"])
        logymax = np.log(stan_data["y"]).max()
        if beta<0 and tau<logymax:
            m += logymax+0.1-tau
        elif beta>0 and tau>logcensor:
            m -= tau-logcensor-0.1

        inits = {\
            "m": m, \
            "logs": math.log(s), \
            "g": g
        }

    else:
        errmsg = f"No initialisation for distribution {dist.name}."
        raise ValueError(errmsg)

    # initialise streamflow error model
    inits["err"] = stan_data["mu_err"]

    return inits


def stan_sample(inits, stan_data, \
                    model, \
                    stan_output_folder, \
                    adapt_delta=None, \
                    nchains=8, \
                    nwarm=10000, \
                    show_progress=False, \
                    show_console=False, \
                    nsamples=20000, **kwargs):
    # Set adapt delta following guidelines
    # diagnose
    if adapt_delta is None:
        if model.name.startswith("gev"):
            adapt_delta = 0.9
        else:
            adapt_delta = 0.8

    # Capture stan messages
    stan_output = model.sample(data=stan_data, \
                        chains=nchains, \
                        iter_warmup=nwarm, \
                        iter_sampling=nsamples//nchains, \
                        output_dir=stan_output_folder, \
                        seed=STAN_SEED, \
                        inits=inits, \
                        adapt_delta=adapt_delta, \
                        show_console=show_console, \
                        show_progress=show_progress, \
                        **kwargs)

    return stan_output


def stan_optimise(inits, stan_data, \
                    model, \
                    stan_output_folder, \
                    algorithm="LBFGS", \
                    show_console=False, \
                    **kwargs):
    # Capture stan messages
    stan_output = model.optimize(data=stan_data, \
                        output_dir=stan_output_folder, \
                        algorithm=algorithm, \
                        seed=STAN_SEED, \
                        inits=inits, \
                        show_console=show_console, \
                        **kwargs)

    return stan_output


def stan_performance(dist, stan_data, stan_output, censor, ppos_cst=0.):
    """ Evaluate stan sampling performance. Compute AIC, BIC, RMSE.

        ppos_cst is the constant used to compute plottiong position.
        Default is set to 0 (unbiased) following RMSFit
        (see 'RMC-BestFit - Quick Start Guide.pdf' page 47)

        censor is a separate argument to avoid using the censor
        value in stan data, which can be set to 0 if the
        censoring process was cancelled.
    """
    # Get obs
    if "z" in stan_data:
        iobs = stan_data["iobs"]
        obs = np.sort(stan_data["y"][iobs-1])
    else:
        obs = np.sort(stan_data["y"])

    # Additional censored data
    icens_perf = obs<censor
    ncens_perf = icens_perf.sum()
    obs = obs[obs>=censor]

    # Remove from nobs the censored values
    # for performance assessment
    nobs = stan_data["Nobs"]-ncens_perf
    ntotal = nobs+stan_data["Ncens"]+ncens_perf

    pobs = sutils.ppos(ntotal, ppos_cst)[-nobs:]
    data = pd.DataFrame({"obs": obs, "pobs": pobs})

    # Get data from stan
    if hasattr(stan_output, "optimized_params_pd"):
        # .. Result of optimisation routine
        df = stan_output.optimized_params_pd
        df.loc[:, "PARAMTYPE"] = "OPTIMISATION"
    else:
        # .. Result of sampling routine
        df = stan_output.draws_pd()
        df.loc[:, "PARAMTYPE"] = "SAMPLING"

        # .. Add mean and median parameters
        dme = pd.DataFrame(df.mean()).T
        dme.loc[:, "PARAMTYPE"] = "SAMPLING_MEAN"
        dmd = pd.DataFrame(df.median()).T
        dmd.loc[:, "PARAMTYPE"] = "SAMPLING_MEDIAN"
        df = pd.concat([df, dme, dmd], ignore_index=True)

    # Initialise score data frame
    scores = pd.DataFrame(np.nan, \
            columns=["RMSE[undef]", "CORR[adim]", \
                        "MAXBIAS[adim]", \
                        "MEANBIAS[adim]", \
                        "MEANRELBIAS[adim]", \
                        "LOGPOST[adim]", \
                        "PARAMTYPE[undef]"], \
            index=df.index)

    # Loop through parameter sets and compute scores
    for idx, row in df.iterrows():
        # Get sim
        dist.set_dict_params(row.to_dict())
        sim = dist.ppf(pobs)
        # Compute performance
        scores.loc[idx, "RMSE[undef]"] = math.sqrt(np.mean((obs-sim)**2))
        scores.loc[idx, "CORR[adim]"] = np.corrcoef(obs, sim)[0, 1]
        scores.loc[idx, "MAXBIAS[adim]"] = (sim[-1]-obs[-1])/obs[-1]
        scores.loc[idx, "MEANBIAS[adim]"] = np.mean((sim-obs)/obs)
        scores.loc[idx, "MEANRELBIAS[adim]"] = np.mean((sim-obs)/(sim+obs))
        scores.loc[idx, "LOGPOST[adim]"] = row["lp__"]
        scores.loc[idx, "PARAMTYPE[undef]"] = row["PARAMTYPE"]

    return scores, data

