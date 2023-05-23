import numpy as np
from pathlib import Path
import fitsio
import matplotlib.pyplot as plt
plt.style.use(['enrique-science'])

def read_catalog(filename, zmin=0.8, zmax=3.5, cap='NGC'):
    """Read a cutsky catalogue and return positions in comoving
    cartesian coordinates."""
    data = fitsio.read(filename)
    print(data.dtype.names)
    Y5_mask = data['STATUS'] & 2**1 != 0
    redshift_mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    if cap == 'NGC':
        cap_mask = (data['RA'] > 80) & (data['RA'] < 300)
    else:
        cap_mask = (data['RA'] < 80) | (data['RA'] > 300)
    mask = Y5_mask & cap_mask & redshift_mask
    ra = data['RA'][mask]
    dec = data['DEC'][mask]
    z = data['Z'][mask]
    nz = data['NZ'][mask]
    return ra, dec, z, nz


samples = ['LRG', 'ELG', 'QSO']
zrange_samples = [
    [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
    [[0.6, 0.8], [0.8, 1.1], [1.1, 1.6]],
    [[0.8, 1.6], [1.6, 2.1], [2.1, 3.5]],

]
root_dir = '/global/homes/e/epaillas/analyse/kp45/bgs/FirstGenMocks/AbacusSummit/CutSky/'
fname_samples = [
    root_dir + 'LRG/z0.800/cutsky_LRG_z0.800_AbacusSummit_base_c000_ph000.fits',
    root_dir + 'ELG/z1.100/cutsky_ELG_z1.100_AbacusSummit_base_c000_ph000.fits',
    root_dir + 'QSO/z1.400/cutsky_QSO_z1.400_AbacusSummit_base_c000_ph000.fits',
]


fig, ax = plt.subplots(1, 4, figsize=(11, 3), sharey=True, layout='constrained')
for i, sample in enumerate(samples):
    
    for zrange in zrange_samples[i]:
        zmin, zmax = zrange
        ra, dec, z, nz = read_catalog(fname_samples[i], zmin=zmin, zmax=zmax, cap='NGC')
        nsub = 10_000
        ax[i].scatter(z[::nsub], 1e4 * nz[::nsub], s=5.0, marker='.')
for aa in ax:
    aa.tick_params(axis='x', labelsize=15)
    aa.tick_params(axis='y', labelsize=15)
ax[0].set_ylabel(r'$n(z)\, [10^4 h^{3}{\rm Mpc^{-3}}]$', fontsize=15)
ax[0].annotate(text='LRG', xy=(0.5, 15), fontsize=15) 
ax[1].annotate(text='ELG', xy=(0.9, 2.5), fontsize=15)
ax[2].annotate(text='QSO', xy=(2.0, 5), fontsize=15)
ax[3].annotate(text='BGS', xy=(0.4, 15), fontsize=15)
fig.supxlabel('redshift ''$z$', fontsize=15)
plt.savefig('fig/png/desi_tracers_nz.png', dpi=500)
