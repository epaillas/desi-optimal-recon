#!/bin/bash

# NOTE: This script is meant to run on a Perlmutter interactive node. After you've
# requested the node, you can simply run it as a bash script: sh Y1_BAOfit.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
STATISTIC=${1:-xi}  # pk or xi
FITTING_METHOD=${2:-minimizer} # minimizer or mcmc
APMODE=${3:-qparqper}  # qperqpar, qisoqap, qiso
BROADBAND=${4:-pcs2}  # pcs, pcs2, power3, power
RECON_ALGORITHM=IFT
RECON_MODE=recsym
JOB_FLAGS="-n 128"
CODE_PATH=/global/u1/e/epaillas/code/optimalrecon/scripts/Y1_BAOfit_$STATISTIC.py
OUTDIR=$HOME"/desi/users/epaillas/Y1/iron/v0.6/blinded/desilike/$FITTING_METHOD/$STATISTIC/"
FIT_FLAGS="--apmode $APMODE --broadband $BROADBAND --fitting_method $FITTING_METHOD --outdir $OUTDIR"
RECON_FLAGS="--recon_algorithm $RECON_ALGORITHM --recon_mode $RECON_MODE"

set -e
# BGS_BRIGHT-21.5 
srun $JOB_FLAGS python $CODE_PATH --tracer BGS_BRIGHT-21.5 --zmin 0.1 --zmax 0.4 $FIT_FLAGS
srun $JOB_FLAGS python $CODE_PATH --tracer BGS_BRIGHT-21.5 --zmin 0.1 --zmax 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 15 $FIT_FLAGS

# LRG_ffa
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.4 --zmax 0.6 $FIT_FLAGS --sigmas 2.0 --sigmapar 9.0 --sigmaper 4.5
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.4 --zmax 0.6 $RECON_FLAGS --smoothing_radius 15 $FIT_FLAGS --sigmas 2.0 --sigmapar 6.0 --sigmaper 3.0
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.6 --zmax 0.8 $FIT_FLAGS --sigmas 2.0 --sigmapar 9.0 --sigmaper 4.5
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.6 --zmax 0.8 $RECON_FLAGS --smoothing_radius 15 $FIT_FLAGS --sigmas 2.0 --sigmapar 6.0 --sigmaper 3.0
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.8 --zmax 1.1 $FIT_FLAGS --sigmas 2.0 --sigmapar 9.0 --sigmaper 4.5
srun $JOB_FLAGS python $CODE_PATH --tracer LRG --zmin 0.8 --zmax 1.1 $VERSION $RECON_FLAGS --smoothing_radius 15 $FIT_FLAGS --sigmas 2.0 --sigmapar 6.0 --sigmaper 3.0

# ELG_LOP_ffa
srun $JOB_FLAGS python $CODE_PATH --tracer ELG_LOPnotqso --zmin 0.8 --zmax 1.1 $FIT_FLAGS --sigmas 2.0 --sigmapar 8.5 --sigmaper 4.5
srun $JOB_FLAGS python $CODE_PATH --tracer ELG_LOPnotqso --zmin 0.8 --zmax 1.1 $RECON_FLAGS --smoothing_radius 15 $FIT_FLAGS --sigmas 2.0 --sigmapar 6.0 --sigmaper 3.0
srun $JOB_FLAGS python $CODE_PATH --tracer ELG_LOPnotqso --zmin 1.1 --zmax 1.6 $FIT_FLAGS --sigmas 2.0 --sigmapar 8.5 --sigmaper 4.5
srun $JOB_FLAGS python $CODE_PATH --tracer ELG_LOPnotqso --zmin 1.1 --zmax 1.6 $RECON_FLAGS --smoothing_radius 15 $FIT_FLAGS --sigmas 2.0 --sigmapar 6.0 --sigmaper 3.0

# QSO_ffa
srun $JOB_FLAGS python $CODE_PATH --tracer QSO_ffa --region GCcomb --zmin 0.8 --zmax 2.1 $FIT_FLAGS --sigmas 2.0 --sigmapar 9.0 --sigmaper 3.5
srun $JOB_FLAGS python $CODE_PATH --tracer QSO_ffa --region GCcomb --zmin 0.8 --zmax 2.1 $RECON_FLAGS --smoothing_radius 30 $FIT_FLAGS --sigmas 2.0 --sigmapar 6.0 --sigmaper 3.0