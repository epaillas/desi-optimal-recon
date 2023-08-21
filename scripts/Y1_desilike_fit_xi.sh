#!/bin/bash
set -e

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
CODE_PATH=/global/u1/e/epaillas/code/optimalrecon/scripts/Y1_desilike_fit_xi.py
JOB_FLAGS="-n 128 -N 1 -C cpu -t 04:00:00 --qos interactive --account desi"

# BGS_BRIGHT-21.5 
srun $JOB_FLAGS python $CODE_PATH --tracer BGS_BRIGHT-21.5 --zmin 0.1 --zmax 0.4 --free_damping --version 0.4 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer BGS_BRIGHT-21.5 --zmin 0.1 --zmax 0.4 --free_damping --version 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 15 --barry_priors

# # LRG
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.4 --zmax 0.6 --free_damping --version 0.4 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.4 --zmax 0.6 --free_damping --version 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.6 --zmax 0.8 --free_damping --version 0.4 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.6 --zmax 0.8 --free_damping --version 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.8 --zmax 1.1 --free_damping --version 0.4 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.8 --zmax 1.1 --free_damping --version 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10 --barry_priors

# ELG_LOPnotqso
srun $JOB_FLAGS python $CODE_PATH --tracer ELG_LOPnotqso --zmin 0.8 --zmax 1.1 --free_damping --version 0.4 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer ELG_LOPnotqso --zmin 0.8 --zmax 1.1 --free_damping --version 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer ELG_LOPnotqso --zmin 1.1 --zmax 1.6 --free_damping --version 0.4 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer ELG_LOPnotqso --zmin 1.1 --zmax 1.6 --free_damping --version 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10 --barry_priors

# QSO
srun $JOB_FLAGS python $CODE_PATH --tracer QSO --zmin 0.8 --zmax 2.1 --free_damping --version 0.4 --barry_priors
srun $JOB_FLAGS python $CODE_PATH --tracer QSO --zmin 0.8 --zmax 2.1 --free_damping --version 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 15 --barry_priors