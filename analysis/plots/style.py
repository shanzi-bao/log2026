"""Shared typography and layout for the six paper figures."""
from __future__ import annotations

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.text import Text

WIDTH = 5.5  # LoG text width in inches.
SIZE = {'title': 9., 'label': 8., 'tick': 7.5, 'legend': 7.5,
        'annotation': 7.5, 'dataset': 8.5}
ORIGINAL_PDF_WIDTHS = [396., 396., 453.6, 411.395, 514.620, 396.]


def configure():
    """Use the paper font family and Matplotlib defaults."""
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": (
                r"\usepackage[T1]{fontenc}"
                r"\usepackage{newtxtext,newtxmath}"
            ),
            "font.size": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def normalize(fig, number, height):
    # Resizing must not ask Matplotlib's automatic locators to change ticks.
    for ax in fig.axes:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_xticks(ax.get_xticks())
        ax.set_yticks(ax.get_yticks())
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    fig.set_size_inches(WIDTH, height)
    # Scale non-text marks to retain their physical sizes at the target width.
    mark_scale = WIDTH * 72 / ORIGINAL_PDF_WIDTHS[number - 1]
    for line in fig.findobj(Line2D):
        line.set_linewidth(line.get_linewidth() * mark_scale)
        line.set_markersize(line.get_markersize() * mark_scale)
        line.set_markeredgewidth(line.get_markeredgewidth() * mark_scale)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_linewidth(coll.get_linewidth() * mark_scale)
            if hasattr(coll, 'get_sizes'):
                coll.set_sizes(coll.get_sizes() * mark_scale ** 2)
        for spine in ax.spines.values():
            spine.set_linewidth(spine.get_linewidth() * mark_scale)
    for text in fig.findobj(Text):
        text.set_fontsize(SIZE['annotation'])
    for ax in fig.axes:
        ax.title.set_fontsize(SIZE['title'])
        ax.xaxis.label.set_fontsize(SIZE['label'])
        ax.yaxis.label.set_fontsize(SIZE['label'])
        ax.tick_params(axis='both', labelsize=SIZE['tick'])
        for legend in [ax.get_legend()]:
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(SIZE['legend'])
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(SIZE['legend'])
    for attr in ['_supxlabel', '_supylabel']:
        if getattr(fig, attr, None) is not None:
            getattr(fig, attr).set_fontsize(SIZE['label'])
    if number == 1:
        # Wrap the third column heading to fit its axis width.
        fig.axes[2].set_title('Input-normalized\ngeometry', fontsize=SIZE['title'], pad=3)
        # The geometry legend must not cover the two curves at the top of
        # the first panel. Give it a dedicated figure-level header strip.
        handles, labels = fig.axes[2].get_legend_handles_labels()
        fig.axes[2].get_legend().remove()
        fig.legend(handles=handles, labels=labels, loc='upper center',
                   bbox_to_anchor=(.5, 1.), ncol=2,
                   fontsize=SIZE['legend'], frameon=False,
                   handlelength=1.6, handletextpad=.45,
                   columnspacing=1.8, borderaxespad=.25)
        fig.subplots_adjust(left=.24, right=.99, bottom=.14, top=.82,
                            hspace=.13, wspace=.32)
        # GridSpec's explicit wspace overrides the subplot defaults.
        fig.axes[0].get_subplotspec().get_gridspec().update(wspace=.32)
        # Keep dataset labels centered against the relocated row axes.
        for row in range(3):
            axis = fig.axes[row * 3]
            center = (axis.get_position().y0 + axis.get_position().y1) / 2
            for text, offset in zip(fig.texts[1 + 3 * row:4 + 3 * row],
                                    [.042, 0., -.042]):
                text.set_y(center + offset)
        for text in fig.texts:
            if text.get_text().startswith(('(a)', '(b)', '(c)')):
                text.set_fontsize(SIZE['dataset'])
        fig._supxlabel.set_text('Message-passing layer')
        # Align the actual top edges of all three titles, reserving the same
        # two-line header area without changing text or shrinking any font.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        first_row = fig.axes[:3]
        title_top = max(ax.title.get_window_extent(renderer).y1 for ax in first_row)
        for ax in first_row:
            top_in_axes = (title_top - ax.bbox.y0) / ax.bbox.height
            ax.set_title(ax.title.get_text(), fontsize=SIZE['title'],
                         y=top_in_axes, pad=0, va='top')
    if number == 5:
        fig.subplots_adjust(left=.105, right=.99, bottom=.17, top=.79,
                            hspace=.18, wspace=.16)
    if number == 3:
        fig.subplots_adjust(left=.11, right=.855, bottom=.11, top=.84,
                            wspace=.20, hspace=.72)
    if number == 4:
        titles = ['(a) First-step entropy\n ',
                  '(b) Final entropy\n' + r'($h=0.865$)',
                  '(c) Final accuracy\n' + r'($h=0.865$)']
        for ax, title in zip(fig.axes, titles):
            ax.set_title(title, fontsize=SIZE['title'], pad=4)
        fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.65,
                            wspace=.44)
    # Automatic per-axis placement otherwise offsets labels whenever tick
    # strings have different widths (e.g. 75 versus 1.0). Matplotlib groups
    # these by subplot column, so different columns remain independent.
    fig.align_ylabels()
    # Use the whole canvas as the consistent centering reference, not the
    # plot-area center (which differs when labels or color bars take space).
    for legend in fig.legends:
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor((.5, anchor.y0), transform=fig.transFigure)
