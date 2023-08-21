#!/bin/bash

# BGS_BRIGHT-21.5
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer BGS_BRIGHT-21.5 --zmin 0.1 --zmax 0.4
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer BGS_BRIGHT-21.5 --zmin 0.1 --zmax 0.4 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 15

# LRG
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer LRG --zmin 0.4 --zmax 0.6
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer LRG --zmin 0.6 --zmax 0.8
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer LRG --zmin 0.8 --zmax 1.1

python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer LRG --zmin 0.4 --zmax 0.6 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer LRG --zmin 0.6 --zmax 0.8 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer LRG --zmin 0.8 --zmax 1.1 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10

# ELG_LOPnotqso
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer ELG_LOPnotqso --zmin 0.8 --zmax 1.1
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer ELG_LOPnotqso --zmin 1.1 --zmax 1.6

python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer ELG_LOPnotqso --zmin 0.8 --zmax 1.1 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer ELG_LOPnotqso --zmin 1.1 --zmax 1.6 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 10

# QSO
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer QSO --zmin 0.8 --zmax 2.1
python Y1_BAO_multipoles.py --free_damping --barry_priors --version 0.4 --tracer QSO --zmin 0.8 --zmax 2.1 --recon_algorithm IFT --recon_mode recsym --smoothing_radius 15
