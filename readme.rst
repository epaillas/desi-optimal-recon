Installation
------------

First, git clone this repo:

If you want any change to the code to take place immediately, either:

1.  Add the "py" directory to your "py" directory to your ``$PYTHONPATH`` environment variable.

2.  Install (and uninstall) the current git checkout:

    $>  python setup.py develop --user

    $>  python setup.py develop --user --uninstall

You can also install a fixed version of the package:

    $>  python setup.py install --user

This code only works on NERSC, and it is recommended to be used with the cosmodesi environment:

    $>  source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

It is also recommended to use the MPI version of pyrecon, which can be loaded also

    $> module swap pyrecon/main pyrecon/mpi

Examples
--------

Reconstruction on QSO cutsky mocks for the NGC region, using random catalogues with 20 times the galaxy number density,
using the IFT algorithm and the recsym convention:

    $> srun -n 128 python recon_cutsky.py --tracer QSO --region NGC --nran 20 --algorithm IFT --convention recsym --outdir my_recon_dir

Power spectrum calculation for the reconstructed catalogues generated above:

    $> srun -n 128 python pk_cutsky.py --tracer QSO --region NGC --nran 20 --algorithm IFT --convention recsym --recon_dir my_recon_dir --outdir my_pk_dir