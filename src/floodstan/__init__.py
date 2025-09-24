import shutil
import re
import os
import warnings
from pathlib import Path
from typing import Callable, Optional

from floodstan.marginals import MARGINAL_NAMES

import cmdstanpy

__version__ = "1.0"

STAN_FILES_FOLDER = Path(__file__).parent / "stan"
CMDSTAN_VERSION = "2.30.1"

# default stan sampling setup
NSAMPLES_DEFAULT = 10000
NCHAINS_DEFAULT = 10
NWARM_DEFAULT = 10000
SEED_DEFAULT = 5446

# Possible method invoked
STAN_METHODS = ["mcmc", "variational", "laplace", "optimize"]

# on Windows specifically, we should point cmdstanpy to the repackaged
# CmdStan if it exists. This lets cmdstanpy handle the TBB path for us.
local_cmdstan = STAN_FILES_FOLDER / f"cmdstan-{CMDSTAN_VERSION}"
if local_cmdstan.exists():
    cmdstanpy.set_cmdstan_path(str(local_cmdstan.resolve()))


def load_stan_model(name: str, method: Optional[str] = "mcmc") -> Callable:
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

    if method not in STAN_METHODS:
        errmess = f"Expected method in '{'/'.join(STAN_METHODS)}'"\
                  + f", got {method}."
        raise ValueError(errmess)

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
        try:
            shutil.copy(
                model.exe_file,  # type: ignore
                STAN_FILES_FOLDER / f"{stan_name}{suffix}",
            )
        except shutil.SameFileError:
            pass

    def fun(*args, **kwargs):
        if method == "mcmc":
            kwargs["show_progress"] = kwargs.get("show_progress", False)

        if "data" not in kwargs:
            errmess = "Expected data argument"
            raise ValueError(errmess)

        # Exclude LogPearson3 from fitting for now
        marginal_codes = [kwargs["data"].get(n, 1)
                          for n in ["ymarginal", "zmarginal"]]
        lp3_code = next(n for k, n in MARGINAL_NAMES.items()
                        if k == "LogPearson3")
        if lp3_code in marginal_codes:
            errmsg = "LogPearson3 is not implemented in stan."
            raise ValueError(errmsg)

        if is_test:
            if "inits" in kwargs:
                errmess = "Expected no inits argument."\
                          + " Supply parameter values through stan_data."
                raise ValueError(errmess)

            # .. specific argument to run a single iteration
            #    of the sampler.
            kwargs["chains"] = 1
            kwargs["seed"] = SEED_DEFAULT
            kwargs["iter_warmup"] = 1
            kwargs["iter_sampling"] = 1
            kwargs["fixed_param"] = True
            kwargs["show_progress"] = False
        else:
            if "inits" not in kwargs:
                errmess = "Expected inits argument"
                raise ValueError(errmess)

            # .. set defaults as per package variables
            kwargs["seed"] = kwargs.get("seed", SEED_DEFAULT)

            if method == "mcmc":
                kwargs["chains"] = kwargs.get("chains", NCHAINS_DEFAULT)
                kwargs["iter_warmup"] = kwargs.get("iter_warmup",
                                                   NWARM_DEFAULT)

                its = kwargs.get("iter_sampling",
                                 NSAMPLES_DEFAULT//NCHAINS_DEFAULT)
                kwargs["iter_sampling"] = its

            elif method == "variational":
                its = kwargs.get("iter_sampling", NSAMPLES_DEFAULT)
                kwargs.pop("iter_sampling")
                kwargs["output_samples"] = its

            elif method == "laplace":
                its = kwargs.get("iter_sampling", NSAMPLES_DEFAULT)
                kwargs.pop("iter_sampling")
                kwargs["draws"] = its

            # Check inits is of the right size
            ninits = len(kwargs["inits"])
            if ninits != 1 and not isinstance(kwargs["inits"], dict):
                if method in ["variational", "laplace", "optimize"]:
                    if ninits != 1:
                        errmess = "Expected 1 initial "\
                                  + f"parameter sets, got {ninits}."
                        raise ValueError(errmess)
                else:
                    if ninits != kwargs["chains"]:
                        nchains = kwargs["chains"]
                        errmess = f"Expected 1 or {nchains} initial "\
                                  + f"parameter sets, got {ninits}."
                        raise ValueError(errmess)

            if method == "laplace":
                inits = kwargs["inits"]
                kwargs.pop("inits")
                kwargs["opt_args"] = {"inits": inits}

        if "output_dir" in kwargs:
            fout = Path(kwargs["output_dir"])
            if not fout.exists():
                errmess = "Output directory does not exist."
                raise ValueError(errmess)

        # the function returns the sample function only,
        # not the full stan model object
        if is_test:
            # .. simplify return for tests
            smp = model.sample(**kwargs)
            return smp.draws_pd().squeeze()
        else:
            if method == "mcmc":
                smp = model.sample(**kwargs)
            elif method == "variational":
                smp = model.variational(**kwargs)
            elif method == "laplace":
                smp = model.laplace_sample(**kwargs)
            elif method == "optimize":
                smp = model.optimize(**kwargs)
            return smp

    return fun


# Stan sampler
univariate_censored_sampling = load_stan_model("univariate_censored_sampling")
bivariate_censored_sampling = load_stan_model("bivariate_censored_sampling")
gls_spatial_sampling = load_stan_model("gls_spatial_sampling")

# Stan test functions
stan_test_marginal = load_stan_model("stan_test_marginal")
stan_test_copula = load_stan_model("stan_test_copula")
stan_test_glsfun = load_stan_model("stan_test_glsfun")
