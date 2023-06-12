import numpy as np
from astropy.table import Table
import os
import logging
from pathlib import Path
from optimalrecon.io_tools import catalog_fn, get_z_cutsky, get_z_cubicbox
from optimalrecon.io_tools import read_positions_weights_cutsky
from pyrecon import utils
from pypower import mpi, setup_logging, CatalogFFTPower
from cosmoprimo.fiducial import DESI
import argparse

logger = logging.getLogger('pk_cutsky')

def run_pk(distance, data_fn, randoms_fn, data_rec_fn, randoms_rec_fn,
    pk_fn, pk_rec_fn=None, boxsize=None, nmesh=None, cellsize=4, nthreads=64,
    dtype='f8', mpicomm=None, **kwargs):
    
    root = mpicomm is None or mpicomm.rank == 0
    if mpicomm is not None:
        pk_kwargs = {'mpicomm': mpicomm, 'mpiroot': 0}
    else:
        pk_kwargs = {}

    if np.ndim(randoms_fn) == 0: randoms_fn = [randoms_fn]
    
    data_positions, data_weights = None, None
    randoms_positions, randoms_weights = None, None
    data_positions_rec, data_weights_rec = None, None
    randoms_positions_rec, randoms_weights_rec = None, None

    if root:
        logger.info('Loading {}.'.format(data_fn))
        (ra, dec, z), data_weights = read_positions_weights_cutsky(data_fn, **kwargs)
        dist = distance(z)
        data_positions = utils.sky_to_cartesian(dist, ra, dec, dtype=dtype)

        logger.info('Loading {}.'.format(randoms_fn))
        (ra, dec, z), randoms_weights = read_positions_weights_cutsky(randoms_fn, **kwargs)
        dist = distance(z)
        randoms_positions = utils.sky_to_cartesian(dist, ra, dec, dtype=dtype)

        if data_rec_fn is not None and randoms_rec_fn is not None:
            logger.info('Loading {}.'.format(data_rec_fn))
            data_rec = Table.read(data_rec_fn)
            (ra, dec, z), data_weights_rec = read_positions_weights_cutsky(data_rec_fn, **kwargs)
            dist = distance(z)
            data_positions_rec = utils.sky_to_cartesian(dist, ra, dec, dtype=dtype)

            logger.info('Loading {}.'.format(randoms_rec_fn))
            (ra, dec, z), randoms_weights_rec = read_positions_weights_cutsky(randoms_rec_fn, **kwargs)
            dist = distance(z)
            randoms_positions_rec = utils.sky_to_cartesian(dist, ra, dec, dtype=dtype)

    kedges = {'step': 0.005}
    result = CatalogFFTPower(
        data_positions1=data_positions,
        data_weights1=data_weights,
        randoms_positions1=randoms_positions,
        randoms_weights1=randoms_weights,
        cellsize=cellsize, resampler='tsc', interlacing=2,
        ells=(0, 2, 4), los='firstpoint', edges=kedges,
        position_type='pos', boxpad=1.2, **pk_kwargs
    )
    poles = result.poles
    k = poles.k
    Pk_0 = poles(ell=0, complex=False)
    Pk_2 = poles(ell=2, complex=False)
    Pk_4 = poles(ell=4, complex=False)
    if root:
        logger.info('Saving {}.'.format(pk_fn))
        utils.mkdir(os.path.dirname(pk_fn))
        cout = {
            'k': k,
            'Pk_0': Pk_0,
            'Pk_2': Pk_2,
            'Pk_4': Pk_4,}
        np.save(pk_fn, cout)

    if data_rec_fn is not None and randoms_rec_fn is not None:
        result = CatalogFFTPower(
            data_positions1=data_positions_rec,
            data_weights1=data_weights_rec,
            shifted_positions1=randoms_positions_rec,
            shifted_weights1=randoms_weights_rec,
            randoms_positions1=randoms_positions,
            randoms_weights1=randoms_weights,
            cellsize=cellsize, resampler='tsc', interlacing=2,
            ells=(0, 2, 4), los='firstpoint', edges=kedges,
            position_type='pos', boxpad=1.2, **pk_kwargs
        )
        poles = result.poles
        k = poles.k
        Pk_0 = poles(ell=0, complex=False)
        Pk_2 = poles(ell=2, complex=False)
        Pk_4 = poles(ell=4, complex=False)
        if root:
            logger.info('Saving {}.'.format(pk_rec_fn))
            utils.mkdir(os.path.dirname(pk_rec_fn))
            cout = {
                'k': k,
                'Pk_0': Pk_0,
                'Pk_2': Pk_2,
                'Pk_4': Pk_4,}
            np.save(pk_rec_fn, cout)

        
        
def get_bias(tracer='ELG'):
    """Get the default tracer bias for a given target sample."""
    if tracer.startswith('ELG'):
        return 1.2
    if tracer.startswith('QSO'):
        return 2.07
    if tracer.startswith('LRG'):
        return 1.99
    if tracer.startswith('BGS'):
        return 1.5
    return 1.2



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tracer', help='tracer to be selected', type=str, default='ELG')
    parser.add_argument('--indir', help='where to find catalogs', type=str, default='/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/')
    parser.add_argument('--recon_dir', help='where to find reconstructed catalogs', type=str, default='/pscratch/sd/e/epaillas/desi/recon_mocks/')
    parser.add_argument('--region', help='regions; by default, run on all regions', type=str, nargs='*', choices=['NGC','SGC'], default=['NGC'])
    parser.add_argument('--zlim', help='z-limits, or options for z-limits, e.g. "highz", "lowz"', type=str, nargs='*', default=None)
    parser.add_argument('--weight_type', help='types of weights to use; "default" just uses WEIGHT column', type=str, default='FKP')
    parser.add_argument('--nran', help='number of random files to combine together (1-18 available)', type=int, default=5)
    parser.add_argument('--nthreads', help='number of threads, not used in case pyrecon/mpi is used (do srun -n 64 python recon.py ... instead)', type=int, default=64)
    parser.add_argument('--outdir',  help='base directory for output (default: SCRATCH)', type=str, default='./')
    parser.add_argument('--algorithm', help='reconstruction algorithm', type=str, choices=['MG', 'IFT', 'IFTP'], default=None)
    parser.add_argument('--convention', help='reconstruction convention', type=str, choices=['reciso', 'recsym'], default='reciso')
    parser.add_argument('--bias', help='bias', type=float, default=None)
    parser.add_argument('--boxsize', help='box size', type=float, default=None)
    parser.add_argument('--nmesh', help='mesh size', type=int, default=None)
    parser.add_argument('--cellsize', help='cell size', type=float, default=7)
    parser.add_argument('--abs_maglim', help='absolute magnitude limit', type=float, default=None)
    parser.add_argument('--app_maglim', help='apparent magnitude limit', type=float, default=None)

    args = parser.parse_args()

    try:
        mpicomm = mpi.COMM_WORLD  # MPI version
    except AttributeError:
        mpicomm = None  # non-MPI version
    root = mpicomm is None or mpicomm.rank == 0
    setup_logging()

    cat_dir = args.indir
    out_dir = args.outdir
    if root: logger.info('Input directory is {}.'.format(cat_dir))
    if root: logger.info('Output directory is {}.'.format(out_dir))

    if args.bias is not None:
        bias = args.bias
    else:
        bias = get_bias(args.tracer)

    regions = args.region

    zbox = get_z_cubicbox(args.tracer)
    if args.zlim is not None:
        zlims = [float(zlim) for zlim in args.zlim]
    else:
        zlims = get_z_cutsky(args.tracer)
    zlims = [(zlims[0], zlims[-1])]

    distance = DESI().comoving_radial_distance 

    for zmin, zmax in zlims:
        for region in regions:
            if root:
                logger.info(f'Running Pk in region {region} in redshift range {(zmin, zmax)} on {mpicomm.size} cores.')
            data_fn = catalog_fn(tracer=args.tracer, mock_type='cutsky')
            randoms_fn = catalog_fn(tracer=args.tracer, mock_type='cutsky', name='randoms', nrandoms=args.nran)
            data_rec_fn, randoms_rec_fn = None, None
            if args.algorithm is not None: 
                data_rec_fn = catalog_fn(tracer=args.tracer, cat_dir=args.recon_dir, rec_type=args.algorithm+args.convention,
                                        name='data', region=region, nrandoms=args.nran, mock_type='cutsky')
                randoms_rec_fn = catalog_fn(tracer=args.tracer, cat_dir=args.recon_dir, rec_type=args.algorithm+args.convention,
                                            name='randoms', region=region, nrandoms=args.nran, mock_type='cutsky')
            pk_fn = os.path.join(args.outdir, f'Pk_cutsky_{args.tracer}_{region}.npy')
            pk_rec_fn = os.path.join(args.outdir, f'Pk_cutsky_{args.tracer}_{region}_{args.algorithm+args.convention}.npy')
            run_pk(distance, data_fn, randoms_fn, data_rec_fn=data_rec_fn, randoms_rec_fn=randoms_rec_fn,
                   pk_fn=pk_fn, pk_rec_fn=pk_rec_fn, boxsize=args.boxsize, nmesh=args.nmesh,
                   cellsize=args.cellsize, dtype='f8', mpicomm=mpicomm, zlim=(zmin, zmax),
                   region=region, weight_type=args.weight_type, abs_maglim=args.abs_maglim,
                   app_maglim=args.app_maglim)