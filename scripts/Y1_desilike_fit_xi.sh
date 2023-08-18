#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

srun -n 128 -N 1 -C cpu -t 04:00:00 --qos interactive --account desi python /global/u1/e/epaillas/code/optimalrecon/scripts/Y1_desilike_fit_xi.py
