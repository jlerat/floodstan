import shutil
import re
import os
import warnings
from pathlib import Path
from typing import Callable

import cmdstanpy

__version__ = "1.0"

STAN_FILES_FOLDER = Path(__file__).parent / "stan"
CMDSTAN_VERSION = "2.30.1"

# default stan sampling setup
NSAMPLES_DEFAULT = 10000
NCHAINS_DEFAULT = 10
NWARM_DEFAULT = 10000
SEED_DEFAULT = 5446


# on Windows specifically, we should point cmdstanpy to the repackaged
# CmdStan if it exists. This lets cmdstanpy handle the TBB path for us.
local_cmdstan = STAN_FILES_FOLDER / f"cmdstan-{CMDSTAN_VERSION}"
if local_cmdstan.exists():
    cmdstanpy.set_cmdstan_path(str(local_cmdstan.resolve()))


def load_stan_model(name: str) -> Callable:
    """
    Try to load precompiled Stan models. If that fails,
    compile them.
    """
    # Get stan executable name
    stan_name = re.sub("^stan_|_sampling$", "", name)

    # Check if the executable is a test
    is_test = bool(re.search("^stan_test_", name))

    # Add exe suffix if we are running on windows
    suffix = ".exe" if os.name == "nt" else ""

    try:
        model = cmdstanpy.CmdStanModel(
            exe_file=STAN_FILES_FOLDER / f"{stan_name}{suffix}",
            stan_file=STAN_FILES_FOLDER / f"{stan_name}.stan"
        )
    except ValueError:
        warnmess = "Failed to load pre-built model "\
                   + f"'{name}{suffix}', compiling"
        warnings.warn(warnmess)

        stan_file = STAN_FILES_FOLDER / f"{stan_name}.stan"
        model = cmdstanpy.CmdStanModel(stan_file=stan_file,
                                       stanc_options={"O1": True})
        shutil.copy(
            model.exe_file,  # type: ignore
            STAN_FILES_FOLDER / f"{stan_name}{suffix}",
        )

    def fun(*args, **kwargs):
        kwargs["show_progress"] = kwargs.get("show_progress", False)

        if "data" not in kwargs:
            errmess = "Expected data argument"
            raise ValueError(errmess)

        if "inits" not in kwargs and not is_test:
            errmess = "Expected inits argument"
            raise ValueError(errmess)

        if is_test:
            # .. specific argument to run a single iteration
            #    of the sampler.
            kwargs["chains"] = 1
            kwargs["seed"] = SEED_DEFAULT
            kwargs["iter_warmup"] = 1
            kwargs["iter_sampling"] = 1
            kwargs["fixed_param"] = True
            kwargs["show_progress"] = False
        else:
            # .. set defaults as per package variables
            kwargs["chains"] = kwargs.get("chains", NCHAINS_DEFAULT)
            kwargs["seed"] = kwargs.get("seed", SEED_DEFAULT)
            kwargs["iter_warmup"] = kwargs.get("iter_warmup", NWARM_DEFAULT)
            its = kwargs.get("iter_sampling",
                             NSAMPLES_DEFAULT//NCHAINS_DEFAULT)
            kwargs["iter_sampling"] = its

            # Check inits is of the right size
            ninits = len(kwargs["inits"])
            if ninits != 1:
                if ninits != kwargs["chains"]:
                    nchains = kwargs["chains"]
                    errmess = f"Expected 1 or {nchains} initial "\
                              + f"parameter sets, got {ninits}."
                    raise ValueError(errmess)

        if "output_dir" in kwargs:
            fout = Path(kwargs["output_dir"])
            if not fout.exists():
                errmess = "Output directory does not exist."
                raise ValueError(errmess)

        # the function returns the sample function only,
        # not the full stan model object
        smp = model.sample(**kwargs)
        if is_test:
            # .. simplify return for tests
            return smp.draws_pd().squeeze()
        else:
            return smp

    return fun


# Stan sampler
univariate_censored_sampling = load_stan_model("univariate_censored_sampling")
bivariate_censored_sampling = load_stan_model("bivariate_censored_sampling")
gls_spatial_sampling = load_stan_model("gls_spatial_sampling")
gls_spatial_generate_sampling = \
    load_stan_model("gls_spatial_generate_sampling")
event_occurrence_sampling = load_stan_model("event_occurrence_sampling")

# Stan test functions
stan_test_marginal = load_stan_model("stan_test_marginal")
stan_test_copula = load_stan_model("stan_test_copula")
stan_test_glsfun = load_stan_model("stan_test_glsfun")
stan_test_discrete = load_stan_model("stan_test_discrete")
