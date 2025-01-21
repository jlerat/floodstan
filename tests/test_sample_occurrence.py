import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, poisson, nbinom, bernoulli

import pytest
import warnings

import importlib
from tqdm import tqdm

from hydrodiy.plot import putils

from floodstan import sample, discretes, \
                event_occurrence_sampling

from test_discretes import sample_from_norm_truncated

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent


def test_stan_discrete_variable(allclose):
    k = poisson(mu=2.5).rvs(size=100)
    msg = "Cannot find"
    with pytest.raises(ValueError, match=msg):
        sv = sample.StanDiscreteVariable(k, "bidule")

    msg = "Expected data"
    with pytest.raises(ValueError, match=msg):
        sv = sample.StanDiscreteVariable(k[:, None], "Poisson")

    kn = k.copy()
    kn[:3] = -1
    msg = "Need all data"
    with pytest.raises(ValueError, match=msg):
        sv = sample.StanDiscreteVariable(kn, "Poisson")

    kn = k.copy()
    kn[:3] = 101
    msg = "Need all data"
    with pytest.raises(ValueError, match=msg):
        sv = sample.StanDiscreteVariable(kn, "Poisson")


    for dname in sample.DISCRETE_NAMES:
        kn = k.clip(0, 1) if dname == "Bernoulli" else k
        sv = sample.StanDiscreteVariable(kn, dname)
        assert allclose(sv.data, kn)
        assert sv.N == len(k)
        assert sv.discrete_code == sample.DISCRETE_NAMES[dname]
        assert sv.discrete_name == dname

        dd = sv.to_dict()
        assert "kdisc" in dd
        assert "k" in dd


def test_occurence_modelling(allclose):
    # Testing occurence modelling following the process described by
    # Samantha R Cook, Andrew Gelman & Donald B Rubin (2006)
    # Validation of Software for Bayesian Models Using Posterior Quantiles,
    # Journal of Computational and Graphical Statistics, 15:3, 675-692,
    # DOI: 10.1198/106186006X136976

    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    plots = {i: n for i, n in enumerate(discretes.DISCRETE_NAMES.keys())}

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 100
    nrepeat = 30
    nrows, ncols = 2, 2
    axwidth, axheight = 5, 5
    print("\n")

    # Setup image
    plt.close("all")
    mosaic = [[plots.get(ncols*ir+ic, ".") for ic in range(ncols)]\
                        for ir in range(nrows)]

    w, h = axwidth*ncols, axheight*nrows
    fig = plt.figure(figsize=(w, h), layout="tight")
    axs = fig.subplot_mosaic(mosaic)

    for dname, ax in axs.items():
        if dname in ["Poisson", "Bernoulli"]:
            parnames = ["locn"]
        else:
            parnames = ["locn", "phi"]

        dist = discretes.factory(dname)

        klocn_prior = [0.5, 1] if dname=="Bernoulli" else [3, 2]
        kphi_prior = [1, 0.5]

        test_stat = []
        # .. double the number of tries to get nrepeat samples
        # .. in case of failure
        for repeat in range(2*nrepeat):
            desc = f"Testing uniform sampling for "+\
                        f"discrete {dname} ({repeat+1}/{nrepeat})"
            print(desc)

            # Generate parameters from prior
            mini, maxi = (0, 1) if dname=="Bernoulli" else (0, 5)
            dist.locn = sample_from_norm_truncated(*klocn_prior, mini, maxi)
            dist.phi = sample_from_norm_truncated(*kphi_prior, 0.1, 2)

            # Generate data from prior params
            N = 100
            ksmp = dist.rvs(N)

            # Configure stan data and initialisation
            try:
                sv = sample.StanDiscreteVariable(ksmp, dname)
            except:
                # Skip if ksmp contains very high unrealistic values
                continue

            stan_data = sv.to_dict()
            stan_data["klocn_prior"] = klocn_prior
            stan_data["kphi_prior"] = kphi_prior

            # Clean output folder
            fout = FTESTS / "sampling" / "occurrence" / dname
            fout.mkdir(parents=True, exist_ok=True)
            for f in fout.glob("*.*"):
                f.unlink()

            # Sample
            try:
                smp = event_occurrence_sampling(\
                    data=stan_data, \
                    chains=4, \
                    seed=SEED, \
                    iter_warmup=5000, \
                    iter_sampling=500, \
                    output_dir=fout, \
                    inits=stan_data)
            except Exception as err:
                continue

            # Get sample data
            df = smp.draws_pd()

            # Test statistic
            Nsmp = len(df)
            test_stat.append([(df.loc[:, f"k{cn}"]<dist[cn]).sum()/Nsmp\
                    for cn in parnames])

            # Stop iterating when the number of samples is met
            if repeat>nrepeat-2:
                break

        test_stat = pd.DataFrame(test_stat, columns=parnames)

        putils.ecdfplot(ax, test_stat)
        ax.axline((0, 0), slope=1, linestyle="--", lw=0.9)

        title = f"{dname} - {len(test_stat)} samples / {nrepeat}"
        ax.set_title(title)

    fp = FTESTS / "images" / f"event_occurence.png"
    fp.parent.mkdir(exist_ok=True)
    fig.savefig(fp)



