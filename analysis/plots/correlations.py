"""Across-setting correlations between diagnostic and accuracy changes."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

from .style import configure

METRICS = [
    ('delta_probe_entropy', 'Probe entropy', 'Probe entropy', r'Entropy change, $\Delta\overline H$'),
    ('delta_dirichlet_relative', 'Scale-normalized Dirichlet energy', 'Dirichlet energy', 'Relative Dirichlet-energy change'),
    ('delta_mad_relative', 'Global MAD', 'Global MAD', 'Relative global MAD change'),
    ('delta_r_group_relative', 'R_Group (relative)', r'$R_{\mathrm{Group}}$', r'Relative $R_{\mathrm{Group}}$ change'),
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_figure() -> plt.Figure:
    """Plot 24 setting means, standard deviations, and linear fits."""
    configure()
    data = Path(__file__).resolve().parents[2] / "results" / "correlations"
    rows = read_csv(data / "setting_changes.csv")
    stats = {r["comparison"]: r for r in read_csv(data / "correlations.csv")}
    assert len(rows) == 24 and all(int(r['runs']) == 5 for r in rows)
    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(6.3, 5.1))
    fig.subplots_adjust(left=.11, right=.855, bottom=.11, top=.84,
                        wspace=.20, hspace=.59)
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=-.05, vmax=.87)
    markers = {'gcn': 'o', 'gat': '^'}
    y = np.asarray([float(r['delta_probe_accuracy_mean']) for r in rows])
    for index, (ax, (column, name, title, xlabel)) in enumerate(zip(axes.flat, METRICS)):
        x = np.asarray([float(r[column + '_mean']) for r in rows])
        r, rho = pearsonr(x, y).statistic, spearmanr(x, y).statistic
        np.testing.assert_allclose([r, rho], [float(stats[name]['pearson_r']),
                                           float(stats[name]['spearman_rho'])], atol=1e-12)
        slope, intercept = np.polyfit(x, y, 1)
        fx = np.linspace(x.min(), x.max(), 200)
        ax.plot(fx, slope * fx + intercept, color='.43', lw=.85,
                linestyle=(0, (4, 2.5)), zorder=1)
        for row, px, py in zip(rows, x, y):
            colour = cmap(norm(float(row['adjusted_homophily'])))
            ax.errorbar(px, py, xerr=float(row[column + '_std']),
                        yerr=float(row['delta_probe_accuracy_std']), fmt='none',
                        ecolor=colour, elinewidth=.75, alpha=.80, capsize=0, zorder=2)
            ax.scatter(px, py, s=18.5, marker=markers[row['architecture']],
                       facecolor=colour, edgecolor='white', linewidth=.35, zorder=3)
        if index == 0:
            positions = {'cora': ('Cora', (-.43, 43)),
                         'amazon-ratings': ('Amazon-Ratings', (-.32, -11)),
                         'roman-empire': ('Roman-Empire', (-.40, -29))}
            for row, px, py in zip(rows, x, y):
                if row['architecture'] == 'gcn' and row['dataset'] in positions:
                    name, pos = positions[row['dataset']]
                    ax.annotate(name, xy=(px, py), xytext=pos, ha='center', va='center',
                                fontsize=7.5, color='.22',
                                arrowprops={'arrowstyle': '-', 'color': '.53',
                                            'linewidth': .45, 'shrinkA': 3, 'shrinkB': 4}, zorder=4)
        ax.axhline(0, color='.66', lw=.6, zorder=0)
        ax.axvline(0, color='.83', lw=.45, zorder=0)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='.89', lw=.4)
        ax.set_ylim(-38, 53)
        ax.set_yticks([-20, 0, 20, 40])
        ax.margins(x=.09)
        ax.set_xlabel(xlabel, labelpad=5)
        ax.set_title(title + '\n' + rf'($r={r:.2f},\ \rho={rho:.2f}$)',
                     pad=9, linespacing=1.15)
        ax.tick_params(axis='y', pad=4)
        if column == 'delta_r_group_relative':
            ax.set_xticks([0, 5, 10, 15, 20, 25])
    fig.supylabel(r'Change in probe accuracy, $\Delta A$ (pp)', x=.027, y=.4875, fontsize=9)
    handles = [Line2D([0], [0], marker=m, ls='none', markerfacecolor='.4',
                      markeredgecolor='white', markeredgewidth=.35, markersize=4.6,
                      label=a.upper()) for a, m in markers.items()]
    handles.append(Line2D([0], [0], color='.43', linestyle=(0, (4, 2.5)), lw=.85,
                          label='Linear fit'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.484, .973),
               ncol=3, handlelength=1.6, columnspacing=1.8, handletextpad=.6)
    cax = fig.add_axes([.894, .24, .013, .49])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label('Adjusted homophily', labelpad=6, fontsize=8.5)
    cbar.set_ticks([0, .2, .4, .6, .8])
    cbar.ax.tick_params(labelsize=7.5, width=.45, length=2.5)
    cbar.outline.set_linewidth(.45)
    return fig
