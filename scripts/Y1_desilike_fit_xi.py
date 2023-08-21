import numpy as np
from pathlib import Path
from desilike.theories.galaxy_clustering import (BAOPowerSpectrumTemplate,
                                                 DampedBAOWigglesTracerCorrelationFunctionMultipoles,)
from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike import setup_logging
from desilike.samplers import EmceeSampler
import argparse
from pathlib import Path


IRON_DIR = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/')
COV_DIR = Path('/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/')

def read_xi_poles(tracer="LRG", region="GCcomb", version="0.4", zmin=0.4, zmax=0.6,
    smin=0, smax=200, recon_algorithm=None, recon_mode='recsym', smoothing_radius=15,
    concatenate=False, ells=[0, 2, 4]):
    if not recon_algorithm:
        data_dir = IRON_DIR / f'v{version}/blinded/xi/smu'
        data_fn = data_dir / f'xipoles_{tracer}_{region}_{zmin}_{zmax}_default_FKP_lin4_njack0_nran4_split20.txt'
    else:
        data_dir = IRON_DIR / f'v{version}/blinded/recon_sm{smoothing_radius}/xi/smu'
        data_fn = data_dir / f'xipoles_{tracer}_{recon_algorithm}{recon_mode}_{region}_{zmin}_{zmax}_default_FKP_lin4_njack0_nran4_split20.txt'
    data = np.genfromtxt(data_fn)
    mask = (data[:, 0] >= smin) & (data[:, 0] <= smax)
    s = data[mask, 0]
    if concatenate:
        poles = np.concatenate([data[mask, 2+ ell//2] for ell in ells])
    else:
        poles = np.array([data[mask, 2+ ell//2] for ell in ells])
    return s, poles

def read_xi_cov(tracer="LRG", region="GCcomb", version="0.4", zmin=0.4, zmax=0.6,
    ells=(0, 2, 4), smin=0, smax=200, recon_algorithm=None, recon_mode='recsym', smoothing_radius=15):
    if not recon_algorithm:
        data_dir = COV_DIR / f'blinded/v0.1/'
        data_fn = data_dir / f'xi024_{tracer}_{region}_{zmin}_{zmax}_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt'
    else:
        data_dir = COV_DIR / f'blinded/v{version}/'
        data_fn = data_dir / f'xi024_{tracer}_{recon_algorithm}{recon_mode}_sm{smoothing_radius}_{region}_{zmin}_{zmax}_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt'
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

# Barry priors
sigma_par = {'BGS_BRIGHT-21.5': {'': {0.1: 11.0}, "recsym": {0.1: 6.5}, "reciso": {0.1: 6.0}},
             'LRG': {'': {0.4: 10.0, 0.6: 9.5, 0.8: 9.0}, "recsym": {0.4: 6.0, 0.6: 5.5, 0.8: 5.0}, "reciso": {0.4: 6.0, 0.6: 5.5, 0.8: 5.0}}, 
              'ELG_LOPnotqso': {'': {0.8: 8.5, 1.1: 8.0}, "recsym": {0.8: 5.5, 1.1: 5.0}, "reciso": {0.8: 5.5, 1.1: 5.0}},
              'QSO': {'': {0.8: 8.0}, "recsym": {0.8: 5.0}, "reciso": {0.8: 5.0}}}

sigma_per = {'BGS_BRIGHT-21.5': {'': {0.1: 6.0}, "recsym": {0.1: 2.0}, "reciso": {0.1: 2.0}},
                'LRG': {'': {0.4: 6.0, 0.6: 5.5, 0.8: 5.0}, "recsym": {0.4: 2.0, 0.6: 2.0, 0.8: 2.0}, "reciso": {0.4: 2.0, 0.6: 2.0, 0.8: 2.0}},
                'ELG_LOPnotqso': {'': {0.8: 5.0, 1.1: 5.0}, "recsym": {0.8: 2.0, 1.1: 1.5}, "reciso": {0.8: 2.0, 1.1: 1.5}},
                'QSO': {'': {0.8: 5.0}, "recsym": {0.8: 1.5}, "reciso": {0.8: 1.5}}}

sigma_s = {'BGS_BRIGHT-21.5': {'': {0.1: 2.0}, "recsym": {0.1: 0.0}, "reciso": {0.1: 0.0}},
           'LRG': {'': {0.4: 0.0, 0.6: 0.0, 0.8: 0.0}, "recsym": {0.4: 0.0, 0.6: 0.0, 0.8: 0.0}, "reciso": {0.4: 0.0, 0.6: 0.0, 0.8: 0.0}},
           'ELG_LOPnotqso': {'': {0.8: 3.0, 1.1: 3.0}, "recsym": {0.8: 0.0, 1.1: 0.0}, "reciso": {0.8: 0.0, 1.1: 0.0}},
           'QSO': {'': {0.8: 3.0}, "recsym": {0.8: 0.0}, "reciso": {0.8: 0.0}}}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tracer', help='tracer to be selected', type=str, default='LRG')
    parser.add_argument('--region', help='regions; by default, run on all regions', type=str, choices=['NGC','SGC', 'GCcomb'], default='GCcomb')
    parser.add_argument('--zlim', help='z-limits, or options for z-limits, e.g. "highz", "lowz"', type=str, nargs='*', default=None)
    parser.add_argument('--version', help='version of the blinded catalogues', type=str, default='0.1')
    parser.add_argument('--zmin', help='minimum redshift', type=float, default=0.4)
    parser.add_argument('--zmax', help='maximum redshift', type=float, default=0.6)
    parser.add_argument('--ells', help='multipoles to be used', type=int, nargs='*', default=[0, 2,])
    parser.add_argument('--recon_algorithm', help='reconstruction method', type=str, default='')
    parser.add_argument('--recon_mode', help='reconstruction convention', type=str, choices=['recsym', 'reciso'], default='')
    parser.add_argument('--smoothing_radius', help='smoothing radius', type=int, default=10)
    parser.add_argument('--debug', help='debug mode', action='store_true')
    parser.add_argument('--free_damping', help='free damping parameters', action='store_true')
    parser.add_argument('--barry_priors', help='use Barry priors', action='store_true')
    parser.add_argument('--save_getdist', help='save chain in getdist format (only works without MPI)', action='store_true')
    args = parser.parse_args()

    setup_logging()

    smin, smax = 20, 200

    s, xi_poles = read_xi_poles(tracer=args.tracer, region=args.region,
                                version=args.version, zmin=args.zmin,
                                zmax=args.zmax, smin=smin, smax=smax,
                                concatenate=True, ells=args.ells,
                                recon_algorithm=args.recon_algorithm,
                                recon_mode=args.recon_mode,
                                smoothing_radius=args.smoothing_radius)

    xi_cov = read_xi_cov(tracer=args.tracer, region=args.region,
                        version=args.version, zmin=args.zmin,
                        zmax=args.zmax, smin=smin, smax=smax,
                        ells=args.ells, recon_algorithm=args.recon_algorithm,
                        recon_mode=args.recon_mode,
                        smoothing_radius=args.smoothing_radius)

    z = (args.zmin + args.zmax) / 2.

    template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI')
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, mode=args.recon_mode,
                                                                 smoothing_radius=args.smoothing_radius)
    observable = TracerCorrelationFunctionMultipolesObservable(data=xi_poles.T, covariance=xi_cov, theory=theory,
                                                              slim={ell: (smin, smax, 4) for ell in args.ells},)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])

    # Analytically solve for broadband parameters (named 'al*_*')
    for param in theory.init.params.select(basename='al*_*'):
        param.update(derived='.auto') 

    if args.free_damping:
        # we let all damping parametes vary freely
        for param in theory.init.params.select(basename='sigma*'):
            param.update(fixed=False)

        if args.barry_priors:   
            mean_sigmapar = sigma_par[args.tracer][args.recon_mode][args.zmin]
            mean_sigmaper = sigma_per[args.tracer][args.recon_mode][args.zmin]
            mean_sigmas = sigma_s[args.tracer][args.recon_mode][args.zmin]

            likelihood.all_params['sigmapar'].update(prior={'dist': 'norm', 'loc': mean_sigmapar, 'scale': 2.0})
            likelihood.all_params['sigmaper'].update(prior={'dist': 'norm', 'loc': mean_sigmaper, 'scale': 2.0})
            likelihood.all_params['sigmas'].update(prior={'dist': 'norm', 'loc': mean_sigmas, 'scale': 2.0})

    if args.debug:
        print('debug')
        # Set damping sigmas to zero for a quick testing
        for param in theory.params.select(basename='sigma*'):
            param.update(value=0., fixed=True) 
        # Fix some broadband parameters (those with k^{-3} and k^{-2}) for a quick testing
        for param in theory.params.select(basename=['al*_-3', 'al*_-2']):
            param.update(value=0., fixed=True)

    eta = 1./3.
    likelihood.all_params['qiso'].update(derived='{qpar}**{eta} * {qper}**(1. - {eta})')
    likelihood.all_params['qap'].update(derived='{qpar} / {qper}')

    print(likelihood.all_params)

    # run MCMC
    output_dir = '/pscratch/sd/e/epaillas/desi/recon_iron/chains'
    flags = f'{args.tracer}_{args.region}_{args.zmin}_{args.zmax}_v{args.version}'
    if args.free_damping:
        flags += '_free_damping'
    if args.barry_priors:
        flags += '_barry_priors'
    if args.recon_algorithm:
        flags += f'_{args.recon_algorithm}{args.recon_mode}_sm{args.smoothing_radius}'
    output_fn = Path(output_dir) / f'{flags}.npy'
    sampler = EmceeSampler(likelihood, nwalkers=64, save_fn=output_fn, seed=42)
    sampler.run(check={'max_eigen_gr': 0.1, 'stable_over': 1})

    if args.save_getdist:
        # also save chain in getdist format
        chain = sampler.chains[0].remove_burnin(0.5)[::10]
        samples = chain.to_getdist()
        chain.write_getdist(output_fn)