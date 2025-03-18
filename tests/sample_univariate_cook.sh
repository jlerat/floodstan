#!/bin/bash -l

set -a 
FHERE=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

conda activate $CONDA_ENV

PYSCRIPT=$FHERE/sample_univariate_cook.py
if [ ! -f "$PYSCRIPT" ]; then
    echo "ERROR - script $PYSCRIPT does not exists"
    exit 1
fi

FLOG=$FHERE/logs/univariate_sampling
mkdir -p $FLOG

# Assuming we run 5 distribution using 2 cpus per process
# => total of 10 cpus
NPARALLEL=2

echo
echo -----------------------
echo CONDA ENV = $CONDA_ENV
echo NPARALLEL = $NPARALLEL
echo PYSCRIPT = $BASE.py
echo -----------------------
echo

echo
echo "$(whereis python)"
echo

# List of marginals
MARGINALS=(
    Gumbel
    GEV 
    LogPearson3
    GeneralizedPareto
    GeneralizedLogistic
    )

# Run
for marginal in "${MARGINALS[@]}"; do
    echo "Running marginal $marginal"
    nohup python $PYSCRIPT -m $marginal -n $NPARALLEL \
        1> $FLOG/univariate_$marginal.log \
        2> $FLOG/univariate_$marginal.err &
done

