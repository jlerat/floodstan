#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : unknown
## Created : 2022-12-02 11:58:57.348818
## Comment : Generate test data for copula  functions from the R pacakge 'copula'
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from hydrodiy.io import csv, iutils

from tqdm import tqdm

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent

fout = froot / "copula_test_data"
fout.mkdir(exist_ok=True)

basename = source_file.stem
LOGGER = iutils.get_logger(basename)


#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------



LOGGER.info("Process completed")

