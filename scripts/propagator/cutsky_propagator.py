import os, sys
import fitsio, asdf
import numpy as np
from scipy import interpolate
from pathlib import Path
from cosmoprimo.fiducial import DESI
from pyrecon import utils
from pyrecon.metrics import MeshFFTCorrelator, MeshFFTPropagator, CatalogMesh
from pypower.mesh import ArrayMesh
from pypower import setup_logging, mpi, MeshFFTPower
import argparse


def fkp_weights(nz, P0=10000):
    return 1 / (1 + nz * P0)

def mask_fun(nz=0, Y5=0, nz_lop=0, Y1=0, Y1BRIGHT=0):
    return nz * (2**0) + Y5 * (2**1) + nz_lop * (2**2) + Y1 * (2**3) + Y1BRIGHT * (2**4)

def interpolate_nz(z, nz, zmin, zmax):
    zbins = np.linspace(zmin, zmax, 41)
    nz_list = []
    for z0, z1 in zip(zbins[0:-1], zbins[1:]):
        zmask = (z>z0)&(z<z1)
        nz_mean = np.mean(nz[zmask])
        nz_list.append(nz_mean)
    
    zmid = (zbins[0:-1]+zbins[1:])/2.0  # middle point for each z bin
    zmid[0] = zbins[0]    # change the first and last point to cover the whole redshift range
    zmid[-1] = zbins[-1]
    nz_array = np.array(nz_list)
    res = interpolate.InterpolatedUnivariateSpline(zmid, nz_array)
    return res
        
    

def main():
    parser = argparse.ArgumentParser(description="Calculate propagator of the cutsky mock.")
    parser.add_argument('--tracer', required=True, type=str, help='tracer name: LRG, ELG_LOP.')
    parser.add_argument('--bias', required=True, type=float, help='tracer bias.')
    parser.add_argument('--zcubic', required=True, type=float, help='the redshift of the cubic box used for the cutsky mock.')
    parser.add_argument('--cap', required=True, type=str, help='sgc, ngc or both, the footprint of DESI.')
    parser.add_argument('--phase', required=True, type=int, help='phase ID.')
    parser.add_argument('--cellsize', required=True, type=float, help='cellsize for the recon, propagator and Pk calculation.')
    parser.add_argument('--input_ic_dir', required=True, type=str, help='input IC directory')
    parser.add_argument('--input_tracer_dir', required=True, type=str, help='input tracer directory')
    parser.add_argument('--output_dir', required=True, type=str, help='output data directory')
    parser.add_argument('--zmin', required=True, type=float, help='the minimum redshift.')
    parser.add_argument('--zmax', required=True, type=float, help='the maximum redshift.')
    parser.add_argument("--add_nzweight", required=True, type=str, help='Add the nz weight for IC or not. True or False.')
    
    args=parser.parse_args()
    
    os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
    os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
    
    setup_logging()
    mpicomm = mpi.COMM_WORLD
    mpiroot = 0
    
    cap = args.cap
    tracer = args.tracer
    cellsize = args.cellsize
    ic_dir = args.input_ic_dir
    data_dir = args.input_tracer_dir
    output_dir = args.output_dir
    zmin, zmax = args.zmin, args.zmax
    add_nzweight = args.add_nzweight
    
    cosmo = DESI()
    node, phase = '000', args.phase
    
    recon_algos = ['IFT']
    #recon_algos = ['IterativeFFT','IterativeFFTParticle']
    #recon_algos = ['MultiGrid', 'IterativeFFT', 'IterativeFFTParticle']
    conventions = ['recsym']
    
    smooth_radii = ['10']
    #smooth_radii = ['10', '15']
    
    los = 'firstpoint'
    bias = args.bias
    
    zcubic = args.zcubic      # the snapshot of a cubic box used for the cutsky mock
    # rescale IC density to low z
    zeff = (zmin + zmax)/2.0
    growth_rate = cosmo.hubble_function(zcubic)/cosmo.hubble_function(zeff) * (1+zeff)/(1+zcubic) * cosmo.growth_rate(zcubic)
    
    ells = (0, 2, 4)
    kedges = np.arange(0.01, 1.0, 0.005)
    muedges = np.linspace(0., 1., 21)

    Nrand = 5

    # calculate Pk from different reconstruction schemes
    for smooth_radius in smooth_radii:
        recon_dir = f"recon_sm{smooth_radius}/"
        for recon_algo in recon_algos:
            if 'IterativeFFT' in recon_algo and phase == 0:
                #niterations = [3, 5, 7]
                niterations = [3]
            else:
                niterations = [3]
                
            for niter in niterations:
            
                for convention in conventions:

                    if mpicomm.rank == mpiroot:
                        positions = {}
                        positions_rec = {}
                        # ----- read pre-recon data
                        data_fn = Path(data_dir, f"{tracer}_ffa_{cap.upper()}_clustering.dat.fits")
                        data = fitsio.read(data_fn)
                        
                        # ----- calculate pre-recon positions of data 
                        zmask = (data['Z']>zmin)&(data['Z']<zmax)
                        dis = cosmo.comoving_radial_distance(data['Z'][zmask])
                        pos = utils.sky_to_cartesian(dis, data['RA'][zmask], data['DEC'][zmask])
                        positions['data'] = pos
                        weights_data_pre = data['WEIGHT'][zmask]
                        
                        # ----- read pre-recon randoms
                        rand_fn = Path(data_dir, f"{tracer}_ffa_{cap.upper()}_0_clustering.ran.fits")
                        randoms = fitsio.read(rand_fn)
                        print(f'Randoms size: {randoms.size}')
                        for rand_id in range(1, Nrand):
                            rand_fn = Path(data_dir, f"{tracer}_ffa_{cap.upper()}_{rand_id}_clustering.ran.fits")
                            rand_sub = fitsio.read(rand_fn)
                            
                            randoms = np.append(randoms, rand_sub) 
                        # ----- calculate post-recon positions of randoms
                        zmask = (randoms['Z']>zmin)&(randoms['Z']<zmax)
                        dis = cosmo.comoving_radial_distance(randoms['Z'][zmask])
                        pos = utils.sky_to_cartesian(dis, randoms['RA'][zmask], randoms['DEC'][zmask])
                        positions['randoms'] = pos     
                        weights_randoms_pre = randoms['WEIGHT'][zmask]
                        
                        # ----- read reconstructed data         
                        data_fn = Path(data_dir, recon_dir, f"{tracer}_ffa_{cap.upper()}_clustering.{recon_algo}{convention}.dat.fits")
                        data = fitsio.read(data_fn)
                        print(f'Data size: {data.size}')
                        
                        # ----- calculate post-recon positions of data 
                        zmask = (data['Z']>zmin)&(data['Z']<zmax)
                        dis = cosmo.comoving_radial_distance(data['Z'][zmask])
                        pos = utils.sky_to_cartesian(dis, data['RA'][zmask], data['DEC'][zmask])
                        positions_rec['data'] = pos
                        weights_data_post = data['WEIGHT'][zmask]

                        # ----- read reconstructed randoms
                        rand_fn = Path(data_dir, recon_dir, f"{tracer}_ffa_{cap.upper()}_0_clustering.{recon_algo}{convention}.ran.fits")
                        randoms = fitsio.read(rand_fn)
                        print(f'Randoms size: {randoms.size}')
                        for rand_id in range(1, Nrand):
                            rand_fn = Path(data_dir, recon_dir, f"{tracer}_ffa_{cap.upper()}_{rand_id}_clustering.{recon_algo}{convention}.ran.fits")
                            rand_sub = fitsio.read(rand_fn)
                            
                            randoms = np.append(randoms, rand_sub)                        
                        
                        # ----- calculate post-recon positions of randoms
                        zmask = (randoms['Z']>zmin)&(randoms['Z']<zmax)
                        dis = cosmo.comoving_radial_distance(randoms['Z'][zmask])
                        pos = utils.sky_to_cartesian(dis, randoms['RA'][zmask], randoms['DEC'][zmask])
                        positions_rec['randoms'] = pos     
                        weights_randoms_post = randoms['WEIGHT'][zmask]

                        #spl_nz_data = interpolate_nz(data['Z'], data['NZ_MAIN'], zmin, zmax)

                        # ------- Read initial condition density field
                        data_fn = Path(ic_dir, f'cutsky_IC_AbacusSummit_base_c000_ph{phase:03d}_Y5_{cap.upper()}_z{zmin:.2f}_{zmax:.2f}.fits') 
                        ic_data = fitsio.read(data_fn)

                        Z_ic  = ic_data['Z_COSMO']
                        zmask = (Z_ic>zmin)&(Z_ic<zmax)
                        
                        ##fp_mask = (ic_data['STATUS'] != 2)&(ic_data['STATUS'] != 18)  # Y1 footprint mask
                        mask_bit = mask_fun(Y1=1)
                        fp_mask = (ic_data['STATUS']&mask_bit == mask_bit)  

                        ic_mask = zmask&fp_mask

                        distance = cosmo.comoving_radial_distance(Z_ic[ic_mask])
                        positions_ic = utils.sky_to_cartesian(distance, ic_data['RA'][ic_mask], ic_data['DEC'][ic_mask])

                        #factor = cosmo.growth_factor(zeff) / cosmo.growth_factor(99.0)
                        data_fn = Path(f"/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/AbacusSummit_base_c000_ph{phase:03d}", 'ic_dens_N576.asdf')
                        with asdf.open(data_fn, lazy_load=False) as af:
                            growth_table = af['header']['GrowthTable']

                        factor = growth_table[zcubic] / growth_table[99.0]
                        print("IC rescale factor:", factor)

                        weights_ic = 1.0 + (ic_data['ONEplusDELTA'][ic_mask] - 1) * factor       # Need to rescale DENSITY 
#                         if add_nzweight == 'True':
#                             """ scaled by the data's n(z); assume the pre-/post-recon n(z) is close to each other """
#                             dndz = spl_nz_data(Z_ic[zmask])
#                             weights_ic *= dndz

#                             weights_ic_random = dndz
#                         else:
                            
                        weights_ic_random = None

                        del(ic_data)

                    else:
                        positions = {'data': None, 'randoms': None}
                        positions_rec = {'data': None, 'randoms': None}
                        weights_data_pre = None
                        weights_randoms_pre = None  
                        weights_data_post = None
                        weights_randoms_post = None  

                        positions_ic = None
                        weights_ic = None
                        weights_ic_random = None


                    rescaled_mesh_ic = CatalogMesh(positions_ic, data_weights=weights_ic, randoms_positions=positions_ic, randoms_weights=weights_ic_random, boxpad=1.5, cellsize=cellsize, resampler='tsc', interlacing=2, position_type='pos', mpicomm=mpicomm, mpiroot=mpiroot)  ## we should consider the weights on the uniform IC positions 

                    # compute correlator/propagator for post-recon
                    ## for an uniform IC, no need to consider "shotnoise"!!
                    power_ic = MeshFFTPower(rescaled_mesh_ic, edges=kedges, ells=ells, los=los, shotnoise=0.)

                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                    output_filename = f'pk_cutsky_IC_{cap.upper()}_c000_ph{phase:03d}_{zmin}z{zmax}_cellsize{cellsize:.1f}.npy'
                    fn = power_ic.mpicomm.bcast(os.path.join(output_dir, output_filename), root=0)
                    fn_txt = power_ic.mpicomm.bcast(os.path.join(output_dir, output_filename.replace('npy', 'txt')), root=0)

                    power_ic.save(fn)
                    power_ic.poles.save_txt(fn_txt, complex=False)
                    power_ic.mpicomm.Barrier()

                    # paint reconstructed positions to mesh
                    # we need to consider the argument of shifted_positions (except for the "rsd" recon convention)
                    mesh_data_recon = CatalogMesh(positions_rec['data'], data_weights=weights_data_post, randoms_positions=positions['randoms'], randoms_weights=weights_randoms_pre, shifted_positions=positions_rec['randoms'], shifted_weights=weights_randoms_post, boxsize=rescaled_mesh_ic.boxsize, boxcenter=rescaled_mesh_ic.boxcenter, cellsize=cellsize, resampler='tsc', interlacing=2, position_type='pos', mpicomm=mpicomm, mpiroot=mpiroot)

                    # compute correlator/propagator
                    correlator_post = MeshFFTCorrelator(mesh_data_recon, rescaled_mesh_ic, edges=(kedges, muedges), los=los)

                    output_filename = f"correlator_{tracer}_{cap.upper()}_c000_ph{phase:03d}_{recon_algo}{convention}_sm{smooth_radius}_{zmin}z{zmax}_cellsize{cellsize:.1f}.npy"
                    fn = correlator_post.num.mpicomm.bcast(os.path.join(output_dir, output_filename), root=0)
                    fn_txt = correlator_post.num.mpicomm.bcast(os.path.join(output_dir, output_filename.replace('npy', 'txt')), root=0)
                    correlator_post.save(fn)
                    correlator_post.save_txt(fn_txt)
                    correlator_post.mpicomm.Barrier()
                
    # for pre-recon result
    # paint pre-reconstructed positions to mesh
    mesh_data_pre = CatalogMesh(positions['data'], data_weights=weights_data_pre, randoms_positions=positions['randoms'], randoms_weights=weights_randoms_pre, boxsize=rescaled_mesh_ic.boxsize, boxcenter=rescaled_mesh_ic.boxcenter, cellsize=cellsize, resampler='tsc', interlacing=2, position_type='pos', mpicomm=mpicomm, mpiroot=mpiroot)

    correlator_pre = MeshFFTCorrelator(mesh_data_pre, rescaled_mesh_ic, edges=(kedges, muedges), los=los)

    output_filename = f"correlator_{tracer}_{cap.upper()}_c000_ph{phase:03d}_pre_recon_{zmin}z{zmax}_cellsize{cellsize:.1f}.npy"
    fn = correlator_pre.num.mpicomm.bcast(os.path.join(output_dir, output_filename), root=0)
    fn_txt = correlator_pre.num.mpicomm.bcast(os.path.join(output_dir, output_filename.replace('npy', 'txt')), root=0)
    
    correlator_pre.save(fn)
    correlator_pre.save_txt(fn_txt)
    correlator_pre.mpicomm.Barrier()
                
if __name__ == '__main__':
    main()

