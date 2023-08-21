from desilike.samples import Chain
from pathlib import Path
from getdist import plots as gdplt
from desilike.samples import plotting
import matplotlib.pyplot as plt

# regions = ['NGC', 'GCcomb']

chains = []
data_dir = '/pscratch/sd/e/epaillas/desi/recon_iron/chains/'
data_fn = data_dir + f'ELG_LOPnotqso_GCcomb_0.8_1.1_v0.4_free_damping.npy'
chain = Chain.load(data_fn)
chain = chain.remove_burnin(0.1)[::10]
chains.append(chain)

g = gdplt.get_subplot_plotter(width_inch=6)
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = ':'
g.settings.axis_marker_color = 'k'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth = 2.0
g.settings.linewidth_contour = 3.0
g.settings.legend_fontsize = 22
g.settings.axes_fontsize = 17
g.settings.axes_labelsize = 22
plotting.plot_triangle(
    chains,
    g=g,
    params=['qpar', 'qper'],
    markers={'qpar': 1., 'qper': 1},
)
plt.show()

# print useful stats
print(chain.to_stats(tablefmt='pretty'))