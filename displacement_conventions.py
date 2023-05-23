from pathlib import Path
import fitsio
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['enrique-science'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

algorithm = 'IterativeFFT'

for convention in ['RecSym', 'RecIso']:
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharey=True, sharex=True, layout='constrained')
    data_dir = Path('/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/BGS/reconstructed_catalogues/galaxies/')
    data_fn = data_dir / f'data_ph000_{algorithm}_{convention}_nmesh512_Rs15.0_f0.636_b1.55.npy'
    data = np.load(data_fn, allow_pickle=True).item()

    disp_x = data['x_rec'] - data['x']
    disp_y = data['y_rec'] - data['y']
    disp_z = data['z_rec'] - data['z']

    mask_x = np.abs(disp_x) < 100
    mask_y = np.abs(disp_y) < 100
    mask_z = np.abs(disp_z) < 100

    ax[0].hist(disp_x[mask_x], bins=100, density=True, histtype='step', color=colors[0], lw=2.0)
    ax[1].hist(disp_y[mask_y], bins=100, density=True, histtype='step', color=colors[0], lw=2.0)
    ax[2].hist(disp_z[mask_z], bins=100, density=True, histtype='step', color=colors[0], lw=2.0)
    
    
    data_dir = Path('/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CubicBox/BGS/reconstructed_catalogues/randoms/')
    data_fn = data_dir / f'randomsX20_ph000_{algorithm}_{convention}_nmesh512_Rs15.0_f0.636_b1.55.npy'
    data = np.load(data_fn, allow_pickle=True).item()

    disp_x = data['x_rec'] - data['x']
    disp_y = data['y_rec'] - data['y']
    disp_z = data['z_rec'] - data['z']

    mask_x = np.abs(disp_x) < 100
    mask_y = np.abs(disp_y) < 100
    mask_z = np.abs(disp_z) < 100

    ax[0].hist(disp_x[mask_x], bins=100, density=True, histtype='step', ls=':', color=colors[2], lw=2.0)
    ax[1].hist(disp_y[mask_y], bins=100, density=True, histtype='step', ls=':', color=colors[2], lw=2.0)
    ax[2].hist(disp_z[mask_z], bins=100, density=True, histtype='step', ls=':', color=colors[2], lw=2.0)

    ax[0].set_xlabel(r'$x_{\rm post} - x_{\rm pre}$', fontsize=15)
    ax[1].set_xlabel(r'$y_{\rm post} - y_{\rm pre}$', fontsize=15)
    ax[2].set_xlabel(r'$z_{\rm post} - z_{\rm pre}$', fontsize=15)

    ax[2].plot(np.nan, np.nan, ls='-', label='galaxies', color=colors[0])
    ax[2].plot(np.nan, np.nan, ls=':', label='randoms', color=colors[2])

    for aa in ax:
        aa.tick_params(axis='x', labelsize=15)
        aa.tick_params(axis='y', labelsize=15)
        aa.set_xlim(-25, 25)

    if convention == 'RecSym':
        leg = ax[2].legend(fontsize=15, handlelength=1.0)

    fig.suptitle(convention, fontsize=15)
    ax[0].set_ylabel('PDF', fontsize=15)
    plt.savefig(f'fig/png/displacement_{convention}.png', dpi=300)
