import math, sys, json
import re
from itertools import product as prod
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from hydrodiy.io import csv
from floodstan import annual_maximum_series as ams

np.random.seed(5446)


@pytest.mark.parametrize("water_year_start",
                         ["jan", "apr", "aug", "nov"])
def test_compute_ams(water_year_start, allclose):
    times = pd.date_range("2000-01-01", "2010-12-31", freq="D")
    se = pd.Series(times.month+np.random.uniform(0, 1, len(times)),
                   index=times)

    df = ams.compute_ams(se, water_year_start=water_year_start)

    for _, ev in df.iterrows():
        s = ev.WATER_YEAR_START
        e = ev.WATER_YEAR_END
        idx = (se.index>=s) & (se.index<=e)
        assert allclose(se.loc[idx].max(), ev.PEAK)

