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
