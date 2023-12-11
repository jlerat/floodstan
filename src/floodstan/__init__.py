import shutil
import warnings
from pathlib import Path

import cmdstanpy

STAN_FILES_FOLDER = Path(__file__).parent / "stan"
CMDSTAN_VERSION = "2.30.1"


# on Windows specifically, we should point cmdstanpy to the repackaged
# CmdStan if it exists. This lets cmdstanpy handle the TBB path for us.
local_cmdstan = STAN_FILES_FOLDER / f"cmdstan-{CMDSTAN_VERSION}"
if local_cmdstan.exists():
    cmdstanpy.set_cmdstan_path(str(local_cmdstan.resolve()))

def load_stan_model(name: str) -> cmdstanpy.CmdStanModel:
    """
    Try to load precompiled Stan models. If that fails,
    compile them.
    """
    try:
        model = cmdstanpy.CmdStanModel(
            exe_file=STAN_FILES_FOLDER / f"{name}.exe",
            stan_file=STAN_FILES_FOLDER / f"{name}.stan",
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

    return model

# Stan sampler
univariate_censored = load_stan_model("univariate_censored")
bivariate_censored = load_stan_model("bivariate_censored")
gls_spatial = load_stan_model("gls_spatial")
gls_spatial_generate = load_stan_model("gls_spatial_generate")
event_occurrence = load_stan_model("event_occurrence")

# Stan test functions
test_marginal = load_stan_model("test_marginal")
test_copula = load_stan_model("test_copula")
test_glsfun = load_stan_model("test_glsfun")
test_discrete = load_stan_model("test_discrete")

