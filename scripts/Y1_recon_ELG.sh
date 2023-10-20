#!/bin/bash

SURVEY="main"
VERSION=0.6
TRACER="ELG_LOPnotqso"
REGION="NGC SGC"
ZLIM="0.8 1.6"
ALGORITHM="IFT"
CONVENTION="recsym"
SMOOTHING_RADIUS=20
GROWTH_RATE=0.900
BIAS=1.2
CELLSIZE=4.0
NRAN=18
INPUT_DIR=/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v"$VERSION"/blinded
OUTPUT_DIR=/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v"$VERSION"/blinded/recon_sm"$SMOOTHING_RADIUS"
CODE_PATH=/global/homes/e/epaillas/code/LSS/scripts/recon.py

source /global/common/software/desi/desi_environment.sh main
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PATH=/global/homes/e/epaillas/code/LSS/bin:$PATH
export PYTHONPATH=/global/homes/e/epaillas/code/LSS/py:$PYTHONPATH
module swap pyrecon/main pyrecon/mpi

JOB_FLAGS="-n 128 -N 1 -C cpu -t 04:00:00 --qos interactive --account desi"

time srun $JOB_FLAGS python $CODE_PATH \
    --tracer $TRACER \
    --survey $SURVEY \
    --zlim $ZLIM \
    --indir $INPUT_DIR \
    --outdir $OUTPUT_DIR \
    --algorithm $ALGORITHM \
    --convention $CONVENTION \
    --f $GROWTH_RATE \
    --bias $BIAS \
    --cellsize $CELLSIZE \
    --smoothing_radius $SMOOTHING_RADIUS \
    --region $REGION \
    --nran $NRAN
