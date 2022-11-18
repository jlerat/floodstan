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


