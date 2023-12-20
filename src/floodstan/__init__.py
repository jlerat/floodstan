import shutil
import re
import warnings
from pathlib import Path
from typing import Callable

import cmdstanpy

STAN_FILES_FOLDER = Path(__file__).parent / "stan"
CMDSTAN_VERSION = "2.30.1"

# default stan sampling setup
NSAMPLES_DEFAULT = 10000
NCHAINS_DEFAULT = 5
NWARM_DEFAULT = 10000
SEED_DEFAULT = 5446 # because that's my number


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

    try:
        model = cmdstanpy.CmdStanModel(
            exe_file=STAN_FILES_FOLDER / f"{stan_name}.exe",
            stan_file=STAN_FILES_FOLDER / f"{stan_name}.stan",
            compile=False,
        )
    except ValueError:
        warnings.warn(f"Failed to load pre-built model '{name}.exe', compiling")
        model = cmdstanpy.CmdStanModel(
            stan_file=STAN_FILES_FOLDER / f"{name}.stan",
            stanc_options={"O1": True},
        )
        shutil.copy(
            model.exe_file,  # type: ignore
            STAN_FILES_FOLDER / f"{name}.exe",
        )

    def fun(*args, **kwargs):
        assert "data" in kwargs, "Expected data argument"
        if is_test:
            # .. specific argument to run a single iteration
            #    of the sampler.
            kwargs["chains"] = 1
            kwargs["seed"] = SEED_DEFAULT
            kwargs["iter_warmup"] = 0
            kwargs["iter_sampling"] = 1
            kwargs["fixed_param"] = True
            kwargs["show_progress"] = False
        else:
            # .. set defaults as per package variables
            kwargs["chains"] = kwargs.get("chains", NCHAINS_DEFAULT)
            kwargs["seed"] = kwargs.get("seed", SEED_DEFAULT)
            kwargs["iter_warmup"] = kwargs.get("iter_warmup", NWARM_DEFAULT)
            kwargs["iter_sampling"] = kwargs.get("iter_sampling", \
                                        NSAMPLES_DEFAULT//NCHAINS_DEFAULT)
            # .. a bit dangerous, but so convenient
            kwargs["inits"] = kwargs.get("inits", kwargs["data"])

        if "output_dir" in kwargs:
            fout = Path(kwargs["output_dir"])
            assert fout.exists(), "Output directory does not exist."

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
gls_spatial_generate_sampling = load_stan_model("gls_spatial_generate_sampling")
event_occurrence_sampling = load_stan_model("event_occurrence_sampling")

# Stan test functions
stan_test_marginal = load_stan_model("stan_test_marginal")
stan_test_copula = load_stan_model("stan_test_copula")
stan_test_glsfun = load_stan_model("stan_test_glsfun")
stan_test_discrete = load_stan_model("stan_test_discrete")

