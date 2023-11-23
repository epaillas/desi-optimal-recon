import numpy as np
from pathlib import Path
from desilike.theories.galaxy_clustering import (BAOPowerSpectrumTemplate,
                                                 DampedBAOWigglesTracerCorrelationFunctionMultipoles,)
from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike import setup_logging
from desilike.profilers import MinuitProfiler
from desilike.samplers import EmceeSampler

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use(['enrique-science.mplstyle',])


IRON_DIR = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/')
COV_DIR = Path('/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/')

def read_xi_poles(tracer="LRG", region="GCcomb", version="0.6", zmin=0.4, zmax=0.6,
    smin=0, smax=200, recon_algorithm=None, recon_mode='recsym', smoothing_radius=15,
    concatenate=False, ells=[0, 2, 4]):
    if not recon_algorithm:
        data_dir = IRON_DIR / f'v{version}/blinded/xi/smu'
        data_fn = data_dir / f'xipoles_{tracer}_{region}_{zmin}_{zmax}_default_FKP_lin4_njack0_nran4_split20.txt'
    else:
        data_dir = IRON_DIR / f'v{version}/blinded/recon_sm{smoothing_radius}/xi/smu'
        data_fn = data_dir / f'xipoles_{tracer}_{recon_algorithm}{recon_mode}_{region}_{zmin}_{zmax}_default_FKP_lin4_njack0_nran4_split20.txt'
    print(f'Reading {data_fn}')
    data = np.genfromtxt(data_fn)
    mask = (data[:, 0] > smin) & (data[:, 0] <= smax)
    s = data[mask, 1]
    if concatenate:
        poles = np.concatenate([data[mask, 2+ ell//2] for ell in ells])
    else:
        poles = np.array([data[mask, 2+ ell//2] for ell in ells])
    return s, poles

def read_xi_cov(tracer="LRG", region="GCcomb", version="0.6", zmin=0.4, zmax=0.6,
    ells=(0, 2, 4), smin=0, smax=200, recon_algorithm=None, recon_mode='recsym', smoothing_radius=15):
    data_dir = COV_DIR / f'blinded/v{version}/'
    if tracer.startswith('LRG'): smoothing_radius = 10
    if tracer.startswith('ELG_LOPnotqso'): smoothing_radius = 10
    if tracer.startswith('QSO'): smoothing_radius = 20
    if not recon_algorithm:
        data_fn = data_dir / f'xi024_{tracer}_{region}_{zmin}_{zmax}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
    else:
        data_fn = data_dir / f'xi024_{tracer}_{recon_algorithm}{recon_mode}_sm{smoothing_radius}_{region}_{zmin}_{zmax}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
    cov = np.genfromtxt(data_fn)
    smid = np.arange(20, 200, 4)
    slim = {ell: (smin, smax) for ell in ells}
    cov = cut_matrix(cov, smid, (0, 2, 4), slim)
    return cov

def cut_matrix(cov, xcov, ellscov, xlim):
    assert len(cov) == len(xcov) * len(ellscov), 'Input matrix has size {}, different than {} x {}'.format(len(cov), len(xcov), len(ellscov))
    indices = []
    for ell, xlim in xlim.items():
        index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
        index = index[(xcov >= xlim[0]) & (xcov <= xlim[1])]
        indices.append(index)
    indices = np.concatenate(indices, axis=0)
    return cov[np.ix_(indices, indices)]

def get_desilike_stats(tracer, version, region, zmin, zmax, smin, smax, ells, recon_algorithm, recon_mode, smoothing_radius, broadband):
    xi_cov = read_xi_cov(tracer=tracer, version=version, region=region,
                        zmin=zmin, zmax=zmax,
                        smin=smin, smax=smax,
                        ells=ells, recon_algorithm=recon_algorithm,
                        recon_mode=recon_mode,
                        smoothing_radius=smoothing_radius)

    s, xi_poles = read_xi_poles(tracer=tracer, version=version, region=region,
                                zmin=zmin, zmax=zmax,
                                smin=smin, smax=smax, concatenate=True, ells=ells,
                                recon_algorithm=recon_algorithm,
                                recon_mode=recon_mode,
                                smoothing_radius=smoothing_radius)

    z = (zmin + zmax) / 2.

    template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI', apmode=apmode,)
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, mode=recon_mode,
                                                                 smoothing_radius=smoothing_radius,
                                                                 broadband=broadband)
    observable = TracerCorrelationFunctionMultipolesObservable(data=xi_poles.T, covariance=xi_cov,
                                                               theory=theory, s=s, ells=ells)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    for param in likelihood.all_params.select(basename=['al*_*', 'bl*_*']):
        if args.fitting_method == 'minimizer':
            param.update(derived='.auto')
        else:
            param.update(derived='.prec')
    if free_damping:
        # we let all damping parametes vary freely
        for param in likelihood.all_params.select(basename='sigma*'):
            param.update(fixed=False)
    else:
        likelihood.all_params['sigmas'].update(fixed=False, prior={'dist': 'norm', 'loc': sigmas, 'scale': 2., 'limits': [0., 20]})
        likelihood.all_params['sigmapar'].update(fixed=False, prior={'dist': 'norm', 'loc': sigmapar, 'scale': 2., 'limits': [0., 20]})
        likelihood.all_params['sigmaper'].update(fixed=False, prior={'dist': 'norm', 'loc': sigmaper, 'scale': 1., 'limits': [0., 20]})

    return theory, observable, likelihood


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracer', help='tracer to be selected', type=str, default='LRG')
    parser.add_argument('--version', help='version of the blinded catalogues', type=str, default='0.6')
    parser.add_argument('--region', help='regions; by default, run on all regions', type=str, choices=['NGC','SGC', 'GCcomb'], default='GCcomb')
    parser.add_argument('--zlim', help='z-limits, or options for z-limits, e.g. "highz", "lowz"', type=str, nargs='*', default=None)
    parser.add_argument('--zmin', help='minimum redshift', type=float, default=0.4)
    parser.add_argument('--zmax', help='maximum redshift', type=float, default=0.6)
    parser.add_argument('--ells', help='multipoles to be used', type=int, nargs='*', default=[0, 2,])
    parser.add_argument('--recon_algorithm', help='reconstruction method', type=str, default='')
    parser.add_argument('--recon_mode', help='reconstruction convention', type=str, choices=['recsym', 'reciso'], default='')
    parser.add_argument('--smoothing_radius', help='smoothing radius', type=int, default=10)
    parser.add_argument('--free_damping', help='free damping parameters', action='store_true')
    parser.add_argument('--apmode', help='AP parametrization', type=str, default='qiso')
    parser.add_argument('--only_now', help='use no-wiggles power spectrum', action='store_true')
    parser.add_argument('--broadband', help='method to model broadband', type=str, default='power')
    parser.add_argument('--sigmas', help='sigma_s', type=float, default=None)
    parser.add_argument('--sigmapar', help='sigma_par', type=float, default=None)
    parser.add_argument('--sigmaper', help='sigma_per', type=float, default=None)
    parser.add_argument('--fitting_method', help='method to fit the data', type=str, default='minimizer')
    parser.add_argument('--outdir', help='output directory', type=str, default=None)
    args = parser.parse_args()

    setup_logging()

    tracer = args.tracer
    version = args.version
    region = args.region
    zmin, zmax = args.zmin, args.zmax
    smin, smax = 50, 150
    free_damping = args.free_damping
    smoothing_radius = args.smoothing_radius
    recon_algorithm = args.recon_algorithm
    recon_mode = args.recon_mode
    apmode = args.apmode
    broadband = args.broadband
    rec = f'_{recon_algorithm}{recon_mode}_sm{smoothing_radius}' if recon_algorithm else ''            
    ells = (0,) if apmode == 'qiso' else (0, 2)

    if not free_damping:
        sigmas = args.sigmas if args.sigmas is not None else 3.0
        if args.sigmapar is not None:
            sigmapar = args.sigmapar
        else:
            sigmapar = 5.0 if recon_algorithm else 10.0
        if args.sigmaper is not None:
            sigmaper = args.sigmaper
        else:
            sigmaper = 3.0 if recon_algorithm else 6.0

    theory, observable, likelihood = get_desilike_stats(tracer, version, region, zmin, zmax, smin, smax, ells,
        recon_algorithm, recon_mode, smoothing_radius, broadband
    )

    output_dir = './' if args.outdir is None else args.outdir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    flags = f'{args.tracer}_{args.region}_{args.zmin}_{args.zmax}_{args.apmode}_{args.broadband}'
    if not args.free_damping:
        flags += f'_sigmas{sigmas}_sigmapar{sigmapar}_sigmaper{sigmaper}'
    if args.recon_algorithm:
        flags += f'_{args.recon_algorithm}{args.recon_mode}_sm{args.smoothing_radius}'
    output_fn = Path(output_dir) / f'{flags}.npy'

    if args.fitting_method == 'minimizer':
        profiler = MinuitProfiler(likelihood, seed=42)
        profiles = profiler.maximize(niterations=10)
        profiles.save(output_fn)
        output_fn = Path(output_dir) / f'{flags}_stats.txt'
        profiles.to_stats(tablefmt='pretty', fn=output_fn)
        likelihood(**profiles.bestfit.choice(input=True))
        if profiler.mpicomm.rank == 0:
            fig = observable.plot()
            ax = fig.axes
            for aa in ax:
                aa.grid(visible=False)
                aa.xaxis.label.set_size(20)
                aa.yaxis.label.set_size(20)
            ax[0].set_title(f'{tracer} {region} 'rf'${zmin} < z < {zmax}$'f' {recon_algorithm}{recon_mode}')
            output_fn = Path(output_dir) / f'{flags}_multipoles.png'
            fig.savefig(output_fn, bbox_inches='tight', dpi=300)
            fig = observable.plot_bao()
            ax = fig.axes
            for aa in ax:
                aa.grid(visible=False)
                aa.xaxis.label.set_size(15)
                aa.yaxis.label.set_size(15)
            ax[0].set_title(f'{tracer} {region} 'rf'${zmin} < z < {zmax}$'f' {recon_algorithm}{recon_mode}')
            output_fn = Path(output_dir) / f'{flags}_BAO.png'
            fig.savefig(output_fn, bbox_inches='tight', dpi=300)
    else:
        sampler = EmceeSampler(likelihood, nwalkers=64, save_fn=output_fn, seed=42)
        chains = sampler.run(min_iterations=200, max_iterations=10000, check={'max_eigen_gr': 0.03})
        if sampler.mpicomm.rank == 0:
            from desilike.samples import Chain, plotting
            chain = Chain.load(output_fn)
            chain = chain.remove_burnin(0.5)[::10]
            if apmode == 'qparqper':
                qiso = (chain['qpar']**(1./3.) * chain['qper']**(2./3.)).clone(param=dict(basename='qiso', derived=True, latex=r'q_{\rm iso}'))
                qap = (chain['qpar'] / chain['qper']).clone(param=dict(basename='qap', derived=True, latex=r'q_{\rm AP}'))
                chain.set(qiso)
                chain.set(qap)
            output_fn = Path(output_dir) / f'{flags}_stats.txt'
            chain.to_stats(tablefmt='pretty', fn=output_fn)
            output_fn = Path(output_dir) / f'{flags}_triangle.png'
            plotting.plot_triangle(chain, fn=output_fn, title_limit=1,
                                params=['qpar', 'qper', 'qiso', 'qap',],
                                markers={'qpar': 1., 'qper': 1, 'qiso': 1, 'qap': 1})