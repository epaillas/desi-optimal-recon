import numpy as np
from pathlib import Path
from desilike.theories.galaxy_clustering import (BAOPowerSpectrumTemplate,
                                                 DampedBAOWigglesTracerCorrelationFunctionMultipoles,)
from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike import setup_logging
from desilike.samples import Chain
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


IRON_DIR = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/')
COV_DIR = Path('/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/')

def read_xi_poles(tracer="LRG", region="GCcomb", version="0.4", zmin=0.4, zmax=0.6,
    smin=0, smax=200, recon=None, smoothing_radius=15, concatenate=False, ells=[0, 2, 4]):
    data_dir = IRON_DIR / f'v{version}/blinded/xi/smu'
    data_fn = data_dir / f'xipoles_{tracer}_{region}_{zmin}_{zmax}_default_FKP_lin4_njack0_nran4_split20.txt'
    data = np.genfromtxt(data_fn)
    mask = (data[:, 0] >= smin) & (data[:, 0] <= smax)
    s = data[mask, 0]
    if concatenate:
        poles = np.concatenate([data[mask, 2+ ell//2] for ell in ells])
    else:
        poles = np.array([data[mask, 2+ ell//2] for ell in ells])
    return s, poles

def read_xi_cov(tracer="LRG", region="GCcomb", version="0.4", zmin=0.4, zmax=0.6, ells=(0, 2, 4), smin=0, smax=200):
    data_dir = COV_DIR / f'blinded/v{version}/'
    data_fn = data_dir / f'xi024_{tracer}_{region}_{zmin}_{zmax}_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt'
    cov = np.genfromtxt(data_fn)
    smid = np.arange(20, 200, 4)
    slim = {ell: (smin, smax) for ell in ells}
    cov = cut_matrix(cov, smid, (0, 2, 4), slim)
    return cov

def cut_matrix(cov, xcov, ellscov, xlim):
    '''
    The function cuts a matrix based on specified indices and returns the resulting submatrix.

    Parameters
    ----------
    cov : 2D array
        A square matrix representing the covariance matrix.
    xcov : 1D array
        x-coordinates in the covariance matrix.
    ellscov : list
        Multipoles in the covariance matrix.
    xlim : tuple
        `xlim` is a dictionary where the keys are `ell` and the values are tuples of two floats
        representing the lower and upper limits of `xcov` for that `ell` value to be returned.

    Returns
    -------
    cov : array
        Subset of the input matrix `cov`, based on `xlim`.
        The subset is determined by selecting rows and columns of `cov` corresponding to the
        values of `ell` and `xcov` that fall within the specified `xlim` range.
    '''
    assert len(cov) == len(xcov) * len(ellscov), 'Input matrix has size {}, different than {} x {}'.format(len(cov), len(xcov), len(ellscov))
    indices = []
    for ell, xlim in xlim.items():
        index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
        index = index[(xcov >= xlim[0]) & (xcov <= xlim[1])]
        indices.append(index)
    indices = np.concatenate(indices, axis=0)
    return cov[np.ix_(indices, indices)]

tracer = 'LRG'
region = 'GCcomb'
version = 0.1
zmin, zmax = 0.4, 0.6
ells = (0, 2)

setup_logging()

smin, smax = 20, 200

s, xi_poles = read_xi_poles(tracer=tracer, region=region,
                        version=version, zmin=zmin,
                        zmax=zmax, smin=smin, smax=smax,
                        concatenate=True, ells=ells)
xi_cov = read_xi_cov(tracer=tracer, region=region,
                 version=version, zmin=zmin,
                 zmax=zmax, smin=smin, smax=smax,
                 ells=ells)
z = (zmin + zmax) / 2.

template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI')
theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template)
observable = TracerCorrelationFunctionMultipolesObservable(data=xi_poles.T, covariance=xi_cov,
                                                    slim={ell: (smin, smax, 4) for ell in ells}, theory=theory)

observable.plot_covariance_matrix()

plt.savefig('fig/Y1_covariance.pdf', bbox_inches='tight')
plt.show()