import json, re, math
import shutil
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import pytest
import warnings

FTESTS = Path(__file__).resolve().parent

def test_readme():
    fread = FTESTS.parent / "README.md"
    with fread.open("r") as fo:
        readme = fo.readlines()

    # Detect code snipets
    reading = False
    snipets = []
    snipet = []
    for line in readme:
        if line.startswith("```python"):
            reading = True

        if line.strip() == "```":
            reading = False
            snipets.append(snipet)
            snipet = []

        if reading:
            if re.search("^figure_file = ", line):
                # Replace image file path
                line = "figure_file = FTESTS / 'test_readme_snipet.png'\n"
            snipet.append(line)

    # Run snipets
    for snipet in snipets:
        code = snipet[1:-1]
        ftest = FTESTS / "test_readme_snipet.py"
        with ftest.open("w") as fo:
            fo.write("from pathlib import Path\n")
            fo.write(f"FTESTS = Path(r'{FTESTS}')\n\n")
            fo.write("def test_snipet():\n")
            for line in code:
                fo.write(" "*4 + line)

        # Check python syntax
        arg = f"--ignore=E302,W291,W293"
        subprocess.run(["flake8", arg, str(ftest)],
                       capture_output=True,
                       check=True)

        # Run test
        subprocess.run(["pytest", str(ftest)],
                       capture_output=True,
                       check=True)

