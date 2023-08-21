import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desilike.samples import Chain
from getdist import plots as gdplt
from pathlib import Path
plt.style.use(['enrique-science'])

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def read_emcee_chain(filename):
    chain = Chain.load(filename)
    chain = chain.remove_burnin(0.1)[::10]
    samples = chain.to_getdist()
    params = samples.getParams()
    samples.addDerived(params.qpar**(1/3) * params.qper**(1-(1/3)), name='qiso', label=r'q_{\rm iso}')
    samples.addDerived(params.qpar / params.qper, name='qap', label=r'q_{\rm AP}')
    return samples


version = 0.4
sm = 10

chain_dir = f'/pscratch/sd/e/epaillas/desi/recon_iron/chains/'


param_names = ['qiso', 'qap', 'qpar', 'qper',]
param_labels = [r'$\alpha_{\rm iso}$', r'$\alpha_{\rm AP}$', r'$\alpha_\parallel$', r'$\alpha_\perp$']
colors = {'BGS_BRIGHT-21.5': '#4477AA',
        'LRG': '#EE6677',
        'ELG_LOPnotqso': '#228833',
        'QSO': '#CCBB44'}

zranges = {
    'BGS_BRIGHT-21.5': [[0.1, 0.4]],
    'LRG': [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
    'ELG_LOPnotqso': [[0.8, 1.1], [1.1, 1.6]],
    'QSO': [[0.8, 2.1]],
}

smoothing_scales = {
    'BGS_BRIGHT-21.5': 15,
    'LRG': 10,
    'ELG_LOPnotqso': 10,
    'QSO': 15,
}

barry_constraints = {
    'BGS_BRIGHT-21.5': {
        0.25: {'qpar': [0.9331, 0.0943], 'qper': [0.9748, 0.0576], 'qiso': [0.9585, 0.0325], 'qap': [0.9632, 0.1292]},
    },
    'LRG': {
        0.5: {'qpar': [0.9007, 0.0216], 'qper': [0.9796, 0.0146], 'qiso': [0.9525, 0.0090], 'qap': [0.9198, 0.0308]},
        0.7: {'qpar': [0.9996, 0.0344], 'qper': [0.9284, 0.0171], 'qiso': [0.9513, 0.0114], 'qap': [1.0773, 0.0502]},
        0.95: {'qpar': [1.0329, 0.0639], 'qper': [0.9631, 0.0298], 'qiso': [0.9782, 0.0078], 'qap': [1.0317, 0.0276]},
    },
    'ELG_LOPnotqso': {
        0.95: {'qpar': [1.0098, 0.1119], 'qper': [0.9209, 0.1083], 'qiso': [0.9454, 0.0588], 'qap': [1.1163, 0.2018]},
        1.35: {'qpar': [1.0189, 0.0342], 'qper': [0.9278, 0.0221], 'qiso': [ 0.9570, 0.0141], 'qap': [1.0992, 0.0541]},
    },
    'QSO': {
        1.45: {'qpar': [0.8644, 0.2354], 'qper': [1.0830, 0.1082], 'qiso': [0.9936, 0.0744], 'qap': [0.8177, 0.2851]},
    },
}

fig, ax = plt.subplots(len(param_names), 1, figsize=(7, 7.5), sharex=True)

for tracer in ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']:
    sm = smoothing_scales[tracer]
    for zrange in zranges[tracer]:
        print(zrange)
        zmin, zmax = zrange
        zmid = float(f'{(zmin + zmax)/2 : .2f}')
        delta = 0.09 if tracer.startswith('ELG') and np.isclose(zmid, 0.95) else 0.0
        color = colors[tracer]

        pre_fn = chain_dir + f'{tracer}_GCcomb_{zmin}_{zmax}_v0.4_free_damping.npy'
        pre_chain = read_emcee_chain(pre_fn)

        post_fn = chain_dir + f'{tracer}_GCcomb_{zmin}_{zmax}_v0.4_free_damping_IFTrecsym_sm{sm}.npy'
        post_chain = read_emcee_chain(post_fn)

        for iparam, param_name in enumerate(param_names):
            pre_param = pre_chain.mean(param_name)
            post_param = post_chain.mean(param_name)

            pre_std = pre_chain.std(param_name)
            post_std = post_chain.std(param_name)

            # print(f'{tracer}, {zmid}')
            barry_param = barry_constraints[tracer][zmid][param_name][0]
            barry_std = barry_constraints[tracer][zmid][param_name][1]

            # print(f'{tracer} {zmin} {zmax} {param_name} {pre_param}+-{pre_std} {post_param}+-{post_std}')
            print(f'{tracer} {zmin} {zmax} {param_name} {(pre_param-1)*100}+-{pre_std*100} {(post_param-1)*100}+-{post_std*100}')

            ax[iparam].errorbar(zmid + delta - 0.03, pre_param, pre_std,
                        color=color, marker='o', ls='', capsize=1.5,
                        elinewidth=1.0, markeredgewidth=1.0, ms=6.0,
                        markerfacecolor='w',
                        markeredgecolor=color)

            ax[iparam].errorbar(zmid + delta + 0.00, post_param, post_std,
                        color=color, marker='o', ls='', capsize=1.5,
                        elinewidth=1.0, markeredgewidth=1.0, ms=6.0,
                        markerfacecolor=lighten_color(color, 0.7),
                        markeredgecolor=color)

            ax[iparam].errorbar(zmid + delta + 0.03, barry_param, barry_std,
                        color=color, marker='o', ls='', capsize=1.5,
                        elinewidth=1.0, markeredgewidth=1.0, ms=6.0,
                        markerfacecolor='k',
                        markeredgecolor=color)

            ax[iparam].set_ylabel(param_labels[iparam], fontsize=18)


ax[-1].set_xlabel('redshift', fontsize=18)
for aa in ax:
    aa.hlines(1.0, 0.1, 1.6, ls='--', color='k', lw=1.0)
    aa.set_xlim(0.1, 1.6)
for aa in ax[:-1]:
    aa.axes.get_xaxis().set_visible(False)
ax[0].plot(np.nan, np.nan, color=colors['BGS_BRIGHT-21.5'], marker='o', ls='', label='BGS_BRIGHT-21.5', ms=6.0)
ax[0].plot(np.nan, np.nan, color=colors['LRG'], marker='o', ls='', label='LRG', ms=6.0)
ax[0].plot(np.nan, np.nan, color=colors['ELG_LOPnotqso'], marker='o', ls='', label='ELG_LOPnotqso', ms=6.0)
ax[0].plot(np.nan, np.nan, color=colors['QSO'], marker='o', ls='', label='QSO', ms=6.0)

# ax[-1].set_ylabel(r'$\chi^2/{\rm dof}$', fontsize=15)
ax[0].legend(bbox_to_anchor=(-0.08, 1.5), loc='upper left',
        frameon=False, fontsize=15, ncols=4, handletextpad=0.0,
        columnspacing=0.3)
plt.tight_layout()
plt.savefig(f'fig/hubble_diagram_desilike.pdf', bbox_inches='tight')
plt.show()


# for iparam, param_name in enumerate(param_names):

#     param = df[f'{param_name}'].values
#     sigma_param = df[f'sigma_{param_name}'].values

#     for i in range(len(zmid)):
#         tracer = names[i].split('BLIND_')[1].split(f'_{cap}')[0]
#         delta = 0.05 if tracer.startswith('ELG') and np.isclose(zmid[i], 0.95) else 0.0
#         color = colors[tracer]
#         ax[iparam].errorbar(zmid[i] + delta, param[i], sigma_param[i],
#                     color=color, marker='o', ls='', capsize=1.5,
#                     elinewidth=1.0, markeredgewidth=1.0, ms=6.0,
#                     markerfacecolor=lighten_color(color, 0.7),
#                     markeredgecolor=color)

#     if param_name == 'epsilon':
#         ax[iparam].hlines(0.0, 0.1, 1.6, ls='--', color='k', lw=1.0)
#     else:
#         ax[iparam].hlines(1.0, 0.1, 1.6, ls='--', color='k', lw=1.0)

#     ax[iparam].set_ylabel(param_labels[iparam], fontsize=18)
#     ax[iparam].tick_params(axis='both', which='major', labelsize=15)
#     ax[iparam].set_xlim(0.17, 1.53)

# for i in range(len(zmid)):
#     tracer = names[i].split('BLIND_')[1].split(f'_{cap}')[0]
#     color = colors[tracer]
#     delta = 0.05 if tracer.startswith('ELG') and np.isclose(zmid[i], 0.95) else 0.0
#     ax[-1].plot(zmid[i] + delta, chi2[i],
#                 color=color, marker='o', ls='',
#                 markeredgewidth=1.0, ms=7.0,
#                 markerfacecolor=lighten_color(color, 0.7),
#                 markeredgecolor=color)

# plt.savefig(f'hubble_diagram_{cap}_v{version}_sm{sm}.png', dpi=300)
# plt.savefig(f'hubble_diagram_{cap}_v{version}_sm{sm}.pdf')
# # plt.show()