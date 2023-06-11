import os
import logging
import numpy as np
from astropy.table import Table, vstack
from LSS.tabulated_cosmo import TabulatedDESI
import matplotlib.pyplot as plt


logger = logging.getLogger('io_tools')


def get_z_cutsky(tracer):
    if tracer.startswith('LRG'):
        return [0.4, 1.1]
    elif tracer.startswith('ELG'):
        return [0.8, 1.6]
    elif tracer.startswith('QSO'):
        return [0.8, 2.1]
    elif tracer.startswith('BGS'):
        return [0.1, 0.4]

def get_z_cubicbox(tracer):
    if tracer.startswith('LRG'):
        return 0.8
    elif tracer.startswith('ELG'):
        return 1.1
    elif tracer.startswith('QSO'):
        return 1.4
    elif tracer.startswith('BGS'):
        return 0.2

def catalog_dir(tracer, mock_type='cutsky', phase_idx=0, cosmo_idx=0, 
    base_dir='/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/'):
    if tracer.startswith('BGS'): tracer += '_v2'
    redshift = get_z_cubicbox(tracer)
    if mock_type == 'cubicbox':
        return os.path.join(base_dir, 'CubicBox', tracer, f'z{redshift:.3f}',
                            f'AbacusSummit_base_c{cosmo_idx:03}_p{phase_idx:03}')
    return os.path.join(base_dir, 'CutSky', tracer, f'z{redshift:.3f}')


def catalog_fn(tracer='LRG', mock_type='cutsky', cat_dir=None, phase=0,
    name='data', nrandoms=4, rec_type=None, region=None, **kwargs):
    if cat_dir is None:
        cat_dir = catalog_dir(tracer=tracer, mock_type=mock_type, **kwargs)
    redshift = get_z_cubicbox(tracer)

    if mock_type == 'cutsky':
        if rec_type:
            if name == 'data':
                return os.path.join(cat_dir, f'cutsky_{tracer}_{region}_z{redshift:.3f}_AbacusSummit_base_c000_ph{phase:03}_{rec_type}.fits')
            return [os.path.join(cat_dir, f'cutsky_{tracer}_{region}_random_S{i*100}_1X_{rec_type}.fits') for i in range(1, nrandoms + 1)]
        if name == 'data':
            return os.path.join(cat_dir, f'cutsky_{tracer}_z{redshift:.3f}_AbacusSummit_base_c000_ph{phase:03}.fits')
        if tracer.startswith('BGS'):
            return [os.path.join(cat_dir, f'random_S{i*100}_1X.fits') for i in range(1, nrandoms + 1)]
        return [os.path.join(cat_dir, f'cutsky_{tracer}_random_S{i*100}_1X.fits') for i in range(1, nrandoms + 1)]
    else:
        raise NotImplementedError(f'catalog_fn not implemented for mock_type={mock_type}')

def read_positions_weights_cubicbox(filename, boxsize, hubble, az, los='z'):
    data = Table.read(filename)
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    x = data['x']
    y = data['y']
    z = data['z']
    x_rsd = x + vx / (hubble * az)
    y_rsd = y + vy / (hubble * az)
    z_rsd = z + vz / (hubble * az)
    x_rsd = x_rsd % boxsize
    y_rsd = y_rsd % boxsize
    z_rsd = z_rsd % boxsize
    weights = np.ones_like(x)
    if los == 'x':
        return x_rsd, y, z, weights
    elif los == 'y':
        return x, y_rsd, z, weights
    elif los == 'z':
        return x, y, z_rsd, weights

def read_positions_weights_cutsky(filename, zlim=None, region='NGC', abs_maglim=None, app_maglim=None,
    weight_type='FKP', return_mask=False):
    if not isinstance(filename, (tuple, list)):
        filename = [filename]
    positions, weights, mask = [], [], []
    for fn in filename:
        data = Table.read(fn)
        if 'LRG' in fn:
            mask_bits = get_desi_mask(main=1, Y5=1)
        else:
            mask_bits = get_desi_mask(nz=1, Y5=0)
        desi_mask = ((data['STATUS'] & (mask_bits)) == mask_bits)
        redshift_mask = (data['Z'] > zlim[0]) & (data['Z'] < zlim[1])
        if region == 'NGC':
            region_mask = (data['RA'] > 80) & (data['RA'] < 300)
        else:
            region_mask = (data['RA'] < 80) | (data['RA'] > 300)
        _mask = desi_mask & redshift_mask & region_mask
        _weights = np.ones_like(data[_mask]['RA'])
        if 'FKP' in weight_type:
            if 'BGS' in fn:
                raise NotImplementedError('FKP weights not implemented for BGS.')
            nz = data[_mask]['NZ']
            _weights /= (1 + nz * 1e4)
        positions.append([data[_mask]['RA'], data[_mask]['DEC'], data[_mask]['Z']])
        weights.append(_weights)
        mask.append(_mask)
    positions = np.concatenate(positions, axis=1)
    weights = np.concatenate(weights)
    mask = np.concatenate(mask)
    if return_mask:
        return positions, weights, mask
    return positions, weights

def get_desi_mask(main=0, nz=0, Y5=0, sv3=0):
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)
