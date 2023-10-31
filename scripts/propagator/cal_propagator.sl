#!/bin/bash
#SBATCH --nodes=4
#SBATCH --account=desi
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --constraint=cpu
##SBATCH -q debug
#SBATCH -q regular
#SBATCH -t 1:40:00
#SBATCH -J prop_cutsky
#SBATCH -o ./stdout/%x.o%j
#SBATCH -e ./stdout/%x.e%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zd585612@ohio.edu

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

module swap pyrecon/main pyrecon/mpi

#cap=sgc
cap_list=(ngc)
#cap_list=(ngc sgc)

#LRG: 0.4<z<0.6, 0.6<z<0.8 or 0.8<z<1.1; zcubic=0.8
#ELG_LOP: 0.8<z<1.1, 1.1<z<1.6; zcubic=1.1

tracer="ELG_LOP"
bias=1.2    # galaxy bias
zcubic=1.1

#zmin=0.8
#zmax=1.1

zmin=1.1
zmax=1.6

cellsize=4

## based on the First-gen mock, adding nz_weight or not does not seem to matter
add_nzweight=False

if [ ${add_nzweight} = "True" ]; then
    dir_nzweight="with_nzweight"
else
    dir_nzweight="no_nzweight"
fi

input_ic_dir="/global/cfs/projectdirs/desi/users/jerryou/MockChallenge/y1_mockchallenge/SecondGenMocks/AbacusSummit/cutsky/IC/"

output_dir="/pscratch/sd/j/jerryou/y1_mockchallenge/SecondGenMocks/propagator/cutsky/${tracer}/"
mkdir -p ${output_dir}


for cap in ${cap_list[*]}; do
  for phase in 0; do
    CAP=$(echo $cap | tr '[:lower:]' '[:upper:]')
    echo $CAP

    input_tracer_dir="/global/cfs/projectdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock${phase}/"

    time srun -n 16 --cpu-bind=cores python cutsky_propagator.py --tracer ${tracer} --cap $cap --bias ${bias} --zcubic ${zcubic} --phase $phase --input_ic_dir ${input_ic_dir} --input_tracer_dir ${input_tracer_dir} --output_dir ${output_dir} --cellsize $cellsize --zmin $zmin --zmax $zmax --add_nzweight ${add_nzweight}

  done
done
