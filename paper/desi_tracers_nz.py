import numpy as np
from pathlib import Path
import fitsio
import healpy as hp
from scipy.interpolate import InterpolatedUnivariateSpline
from cosmoprimo.fiducial import DESI
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
        region_mask = (data['RA'] > 80) & (data['RA'] < 300)
    else:
        region_mask = (data['RA'] < 80) | (data['RA'] > 300)
    if 'BGS' in filename:
        m_r = data['R_MAG_APP']
        M_r = data['R_MAG_ABS'] - E_correction(data['Z'])
        magnitude_mask = (M_r < -21.5) & (m_r < 19.5)
        mask = Y5_mask & region_mask & redshift_mask & magnitude_mask
    else:
        mask = Y5_mask & region_mask & redshift_mask
    ra = data['RA'][mask]
    dec = data['DEC'][mask]
    z = data['Z'][mask]
    if 'NZ' in data.dtype.names:
        nz = data['NZ'][mask]
    else:
        cosmo = DESI()
        nz = get_nofz(ra, dec, z, cosmo, zmin, zmax)
    return ra, dec, z, nz

def E_correction(z, Q0=-0.97, z0=0.1):
    return Q0 * (z - z0)

def get_nofz(ra, dec, z, cosmo, zmin, zmax):
    nside = 256
    npix = hp.nside2npix(nside)
    phi = np.radians(ra)
    theta = np.radians(90.0 - dec)
    pixel_indices = hp.ang2pix(nside, theta, phi)
    pixel_unique = np.unique(pixel_indices)
    fsky = len(pixel_unique)/npix
    spl_nz = spl_nofz(z, fsky, cosmo, zmin, zmax)
    nz = spl_nz(z)
    return nz

def spl_nofz(zarray, fsky, cosmo, zmin, zmax, Nzbins=100):
    zbins = np.linspace(zmin, zmax, Nzbins+1)
    Nz, zbins = np.histogram(zarray, zbins)
    zmid = zbins[0:-1] + (zmax-zmin)/Nzbins/2.0
    zmid[0], zmid[-1] = zbins[0], zbins[-1]
    rmin = cosmo.comoving_radial_distance(zbins[0:-1])
    rmax = cosmo.comoving_radial_distance(zbins[1:])
    vol = fsky * 4./3*np.pi * (rmax**3.0 - rmin**3.0)
    nz_array = Nz/vol
    spl_nz = InterpolatedUnivariateSpline(zmid, nz_array)
    return spl_nz


samples = ['BGS', 'LRG', 'ELG', 'QSO']
zrange_samples = [
    [[0.1, 0.4]],
    [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
    [[0.6, 0.8], [0.8, 1.1], [1.1, 1.6]],
    [[0.8, 1.6], [1.6, 2.1], [2.1, 3.5]],

]
root_dir = '/global/homes/e/epaillas/analyse/kp45/bgs/FirstGenMocks/AbacusSummit/CutSky/'
fname_samples = [
    root_dir + 'BGS_v2/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits',
    root_dir + 'LRG/z0.800/cutsky_LRG_z0.800_AbacusSummit_base_c000_ph000.fits',
    root_dir + 'ELG/z1.100/cutsky_ELG_z1.100_AbacusSummit_base_c000_ph000.fits',
    root_dir + 'QSO/z1.400/cutsky_QSO_z1.400_AbacusSummit_base_c000_ph000.fits',
]


fig, ax = plt.subplots(1, 4, figsize=(11, 3), sharey=True, layout='constrained')
for i, sample in enumerate(samples):
    
    for zrange in zrange_samples[i]:
        zmin, zmax = zrange
        ra, dec, z, nz = read_catalog(fname_samples[i], zmin=zmin, zmax=zmax, cap='NGC')
        if 'BGS' in fname_samples[i]:
            nsub = 100
        else:
            nsub = 10_000
        ax[i].scatter(z[::nsub], 1e4 * nz[::nsub], s=5.0, marker='.')
for aa in ax:
    aa.tick_params(axis='x', labelsize=15)
    aa.tick_params(axis='y', labelsize=15)
ax[0].set_ylabel(r'$n(z)\, [10^4 h^{3}{\rm Mpc^{-3}}]$', fontsize=15)
ax[0].set_title('BGS', fontsize=15)
ax[1].set_title('LRG', fontsize=15) 
ax[2].set_title('ELG', fontsize=15)
ax[3].set_title('QSO', fontsize=15)
fig.supxlabel('redshift ''$z$', fontsize=15)
plt.savefig('fig/png/desi_tracers_nz.png', dpi=500)
