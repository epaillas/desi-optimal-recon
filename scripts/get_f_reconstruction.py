import numpy as np
import fitsio
import argparse
from pathlib import Path
from cosmoprimo.fiducial import DESI

def read_catalog(filename, zmin, zmax, weight_type='default'):
    data = fitsio.read(filename)
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    redshift = data[mask]['Z']
    weights = np.ones_like(redshift)
    if 'default' in weight_type:
        weights *= data[mask]['WEIGHT']
    if 'FKP' in weight_type:
        weights *= data[mask]['WEIGHT_FKP']
    return redshift, weights


parser = argparse.ArgumentParser()
parser.add_argument('--tracer', help='tracer to be selected', type=str, default='LRG')
parser.add_argument('--zlim', help='z-limits, or options for z-limits, e.g. "highz", "lowz"', type=str, nargs='*', default=None)
parser.add_argument('--region', help='regions; by default, run on all regions', type=str, nargs='*', choices=['NGC','SGC','N', 'S', 'DN', 'DS', ''], default=['NGC'])
parser.add_argument('--weight_type', help='types of weights to use; "default" just uses WEIGHT column', type=str, default='default_FKP')
parser.add_argument('--version', help='version of the LSS catalog', type=str, default='v0.6.1')

args = parser.parse_args()

if args.zlim is None:
    if 'QSO' in args.tracer:
        zlims = [0.8, 2.1]
    elif 'ELG' in args.tracer:
        zlims = [0.8, 1.6]
    elif 'LRG' in args.tracer:
        zlims = [0.4, 1.1]
    elif 'BGS' in args.tracer:
        zlims = [0.1, 0.4]
else:
    zlims = [float(zlim) for zlim in args.zlim]

cosmo = DESI()

data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{args.version}/blinded'

for region in args.region:
    data_fn  = Path(data_dir) / f'{args.tracer}_{region}_clustering.dat.fits'
    redshift, weights = read_catalog(data_fn, zmin=zlims[0], zmax=zlims[1], weight_type=args.weight_type)

    effective_redshift = np.average(redshift, weights=weights)
    f = cosmo.sigma8_z(z=effective_redshift, of='theta_cb') / cosmo.sigma8_z(z=effective_redshift, of='delta_cb')

    print(f'{args.tracer} {region}')
    print(f'zlim = {zlims}')
    print(f'effz = {effective_redshift:.3f}')
    print(f'f = {f:.3f}')
