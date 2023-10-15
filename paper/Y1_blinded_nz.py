#!/usr/bin/env python
# coding: utf-8

## Plot Y1 data v0.6 n(z). --08-31-2023


#%pylab inline
import numpy as np
import matplotlib.pyplot as plt


samples = ['BGS', 'LRG', 'ELG', 'QSO']
zrange_samples = [
    [[0.1, 0.4]],
    [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
    [[0.8, 1.1], [1.1, 1.6]], 
    [[0.8, 1.6], [1.6, 2.1], [2.1, 3.5]],

] # for ELG, there is no redshift range [0.6, 0.8]
##cap = "NGC"
cap_list = ["NGC", "SGC"]
line_list = ["-", "--"]



idir = "/global/cfs/projectdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/"
filename_list = ["BGS_BRIGHT-21.5_{cap}_nz.txt", "LRG_{cap}_nz.txt", "ELG_LOPnotqso_{cap}_nz.txt", "QSO_{cap}_nz.txt"]



fig, ax = plt.subplots(1, 4, figsize=(14, 4), sharey=True, layout='constrained')

for cap, ls in zip(cap_list, line_list):
    for i, filename in enumerate(filename_list):
        ifile = idir + filename
        zmid, zlow, zhigh, nz, Nbin, Vol_bin = np.loadtxt(ifile.format(cap=cap), unpack=True)
        for j, zrange in enumerate(zrange_samples[i]):
            zmin, zmax = zrange[0], zrange[1]
            #print(zmin, zmax)
            zmask = (zmid>=zmin)&(zmid<zmax)

            ax[i].plot(zmid[zmask], 1e4 * nz[zmask], lw=2.0, ls=ls, color=f'C{j}', label=f"{cap}" if i==0 else None)

        for i, aa in enumerate(ax):
            aa.tick_params(axis='x', labelsize=15)
            aa.tick_params(axis='y', labelsize=15)
            aa.set_title(samples[i], fontsize=15)
            
ax[0].legend(fontsize=15)
ax[0].set_ylabel(r'$n(z)\, [10^4 h^{3}{\rm Mpc^{-3}}]$', fontsize=20)

fig.supxlabel('redshift ''$z$', fontsize=20)
plt.savefig('figs/Y1_v0.6_nz.png', dpi=500)



