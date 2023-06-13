import numpy as np
from pathlib import Path
from pypower import CatalogFFTPower
import matplotlib.pyplot as plt
plt.style.use(['enrique-science'])

# BGS
multipoles_phases = []
for phase in range(25):
    data_dir = Path('/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/BGS/Pk/Pre/epaillas')
    data_fn = data_dir / f'BGS_PreRecon_Pk_Mr-21.5_mr19.5_zmin0.1_zmax0.4_ph{phase:03}.npy'
    result = CatalogFFTPower.load(data_fn)
    poles = result.poles[::5]
    poles.select((0, 0.3))
    k_bgs = poles.k
    Pk_0 = poles(ell=0, complex=False)
    Pk_2 = poles(ell=2, complex=False)
    Pk_4 = poles(ell=4, complex=False)
    multipoles_phases.append([Pk_0, Pk_2, Pk_4])
multipoles_phases = np.asarray(multipoles_phases)
multipoles_bgs = multipoles_phases.mean(axis=0)
err_bgs = multipoles_phases.std(axis=0)

# LRG
multipoles_phases = []
for phase in range(25):
    data_dir = Path('/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/LRG/Pk/Pre/zhejie/NGC')
    data_fn = data_dir / f'cutsky_LRG_NGC_0.8z1.1_ph{phase:03}.randoms20X.pre_recon_redshiftspace_pad1.5_Pk_nmesh1024.npy'
    result = CatalogFFTPower.load(data_fn)
    poles = result.poles
    poles.select((0, 0.3))
    k_lrg = poles.k
    Pk_0 = poles(ell=0, complex=False)
    Pk_2 = poles(ell=2, complex=False)
    Pk_4 = poles(ell=4, complex=False)
    multipoles_phases.append([Pk_0, Pk_2, Pk_4])
multipoles_phases = np.asarray(multipoles_phases)
multipoles_lrg = multipoles_phases.mean(axis=0)
err_lrg = multipoles_phases.std(axis=0)

# ELG
multipoles_phases = []
for phase in range(25):
    data_dir = Path('/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/ELG/Xinyi/cellsize4/ELG_metrics/Pk/pknmesh1024/NGC')
    data_fn = data_dir / f'cutsky_ELG_NGC_1.1z1.6_ph{phase:03}.randoms20X.pre_recon_redshiftspace_pad1.5_Pk_nmesh1024.npy'
    result = CatalogFFTPower.load(data_fn)
    poles = result.poles
    poles.select((0, 0.3))
    k_elg = poles.k
    Pk_0 = poles(ell=0, complex=False)
    Pk_2 = poles(ell=2, complex=False)
    Pk_4 = poles(ell=4, complex=False)
    multipoles_phases.append([Pk_0, Pk_2, Pk_4])
multipoles_phases = np.asarray(multipoles_phases)
multipoles_elg = multipoles_phases.mean(axis=0)
err_elg = multipoles_phases.std(axis=0)

# QSO
multipoles_phases = []
for phase in range(25):
    data_dir = Path('/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/AbacusSummit/CutSky/QSO/Pk/Pre/epaillas/')
    data_fn = data_dir / f'QSO_PreRecon_Pk_zmin0.8_zmax1.6_ph{phase:03}.npy'
    result = CatalogFFTPower.load(data_fn)
    poles = result.poles[::5]
    poles.select((0, 0.3))
    k_qso = poles.k
    Pk_0 = poles(ell=0, complex=False)
    Pk_2 = poles(ell=2, complex=False)
    Pk_4 = poles(ell=4, complex=False)
    multipoles_phases.append([Pk_0, Pk_2, Pk_4])
multipoles_phases = np.asarray(multipoles_phases)
multipoles_qso = multipoles_phases.mean(axis=0)
err_qso = multipoles_phases.std(axis=0)

fig, ax = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True, layout='constrained')
phases = list(range(0, 25))

ax[0].plot(k_bgs, k_bgs*multipoles_bgs[0], label='$\ell = 0$')
ax[0].plot(k_bgs, k_bgs*multipoles_bgs[1], label='$\ell = 2$')
ax[0].plot(k_bgs, k_bgs*multipoles_bgs[2], label='$\ell = 4$')
ax[0].fill_between(k_bgs, k_bgs*(multipoles_bgs[0] - err_bgs[0]), k_bgs*(multipoles_bgs[0] + err_bgs[0]), alpha=0.3)
ax[0].fill_between(k_bgs, k_bgs*(multipoles_bgs[1] - err_bgs[1]), k_bgs*(multipoles_bgs[1] + err_bgs[1]), alpha=0.3)
ax[0].fill_between(k_bgs, k_bgs*(multipoles_bgs[2] - err_bgs[2]), k_bgs*(multipoles_bgs[2] + err_bgs[2]), alpha=0.3)

ax[1].plot(k_lrg, k_lrg*multipoles_lrg[0], label='$\ell = 0$')
ax[1].plot(k_lrg, k_lrg*multipoles_lrg[1], label='$\ell = 2$')
ax[1].plot(k_lrg, k_lrg*multipoles_lrg[2], label='$\ell = 4$')
ax[1].fill_between(k_lrg, k_lrg*(multipoles_lrg[0] - err_lrg[0]), k_lrg*(multipoles_lrg[0] + err_lrg[0]), alpha=0.3)
ax[1].fill_between(k_lrg, k_lrg*(multipoles_lrg[1] - err_lrg[1]), k_lrg*(multipoles_lrg[1] + err_lrg[1]), alpha=0.3)
ax[1].fill_between(k_lrg, k_lrg*(multipoles_lrg[2] - err_lrg[2]), k_lrg*(multipoles_lrg[2] + err_lrg[2]), alpha=0.3)

ax[2].plot(k_elg, k_elg*multipoles_elg[0], label='$\ell = 0$')
ax[2].plot(k_elg, k_elg*multipoles_elg[1], label='$\ell = 2$')
ax[2].plot(k_elg, k_elg*multipoles_elg[2], label='$\ell = 4$')
ax[2].fill_between(k_elg, k_elg*(multipoles_elg[0] - err_elg[0]), k_lrg*(multipoles_elg[0] + err_elg[0]), alpha=0.3)
ax[2].fill_between(k_elg, k_elg*(multipoles_elg[1] - err_elg[1]), k_lrg*(multipoles_elg[1] + err_elg[1]), alpha=0.3)
ax[2].fill_between(k_elg, k_elg*(multipoles_elg[2] - err_elg[2]), k_lrg*(multipoles_elg[2] + err_elg[2]), alpha=0.3)

ax[3].plot(k_qso, k_qso*multipoles_qso[0], label='$\ell = 0$')
ax[3].plot(k_qso, k_qso*multipoles_qso[1], label='$\ell = 2$')
ax[3].plot(k_qso, k_qso*multipoles_qso[2], label='$\ell = 4$')
ax[3].fill_between(k_qso, k_qso*(multipoles_qso[0] - err_qso[0]), k_qso*(multipoles_qso[0] + err_qso[0]), alpha=0.3)
ax[3].fill_between(k_qso, k_qso*(multipoles_qso[1] - err_qso[1]), k_qso*(multipoles_qso[1] + err_qso[1]), alpha=0.3)
ax[3].fill_between(k_qso, k_qso*(multipoles_qso[2] - err_qso[2]), k_qso*(multipoles_qso[2] + err_qso[2]), alpha=0.3)

    
ax[0].set_ylabel(r'$kP_{\ell}(k)\,[h^2{\rm Mpc^{-2}}]$', fontsize=15)
leg = ax[0].legend(handlelength=0.0, labelspacing=0.1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
fig.supxlabel(r'$k\,[h{\rm Mpc^{-1}}]$', fontsize=15)
for aa in ax:
    aa.tick_params(axis='x', labelsize=15)
    aa.tick_params(axis='y', labelsize=15)

titles = ['BGS', 'LRG', 'ELG', 'QSO']
for i, aa in enumerate(ax):
    aa.set_title(titles[i], fontsize=15)
plt.savefig('fig/png/desi_tracers_multipoles.png', dpi=500)
