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

from nrivfloodfreqstan import load_stan_model


# Path to stan models
FSTANFILES = Path(__file__).resolve().parent / "stan"
FSTANEXE = FSTANFILES.parent / "stan_executables"
FSTANEXE.mkdir(exist_ok=True)

# Setup logger with a write function to use contextlib
LOGGER = logging.getLogger("cmdstanpy")

# List of stan model
STAN_MODEL_NAMES = ["bivariate_censoring", \
                        "univariate_censoring", \
                        "test_stan_functions"]

STAN_SEED = 5446


def get_stan_model(stan_model_name):
    if stan_model_name.startswith("test"):
        return load_stan_model("test")
    else:
        return load_stan_model(stan_model_name)


def stan_sample(inits, stan_data, \
                    model, \
                    stan_output_folder, \
                    adapt_delta=None, \
                    nchains=5, \
                    nwarm=10000, \
                    show_progress=False, \
                    show_console=False, \
                    nsamples=10000, **kwargs):

    # Add default arguments
    return model.sample(data=stan_data, \
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


