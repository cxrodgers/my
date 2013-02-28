import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def font_embed():
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['svg.fonttype'] = 'svgfont'

def rescue_tick(ax=None, f=None, x=3, y=3):
    # Determine what axes to process
    if ax is not None:
        ax_l = [ax]
    elif f is not None:
        ax_l = f.axes
    else:
        raise ValueError("either ax or f must not be None")
    
    # Iterate over axes to process
    for ax in ax_l:
        if x is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(x))
        if y is not None:
            ax.yaxis.set_major_locator(plt.MaxNLocator(y))
        

def scatter_with_trend(x, y, xname='X', yname='Y', ax=None, 
    legend_font_size='medium'):
    """Scatter plot `y` vs `x`, also linear regression line"""
    dropna = np.isnan(x) | np.isnan(y)
    x = x[~dropna]
    y = y[~dropna]
    
    if ax is None:    
        f = plt.figure()
        ax = f.add_subplot(111)
    ax.plot(x, y, '.')

    m, b, rval, pval, stderr = \
        scipy.stats.stats.linregress(x.flatten(), y.flatten())
    
    trend_line_label = 'r=%0.3f p=%0.3f' % (rval, pval)
    ax.plot([x.min(), x.max()], m * np.array([x.min(), x.max()]) + b, 'k:',
        label=trend_line_label)
    ax.legend(loc='best', prop={'size':legend_font_size})
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    plt.show()

def horiz_bar(bar_lengths, bar_labels=None, bar_positions=None, ax=None,
    bar_errs=None, bar_colors=None, bar_hatches=None):
    """Horizontal bar plot"""
    # Defaults
    if bar_positions is None:
        bar_positions = list(range(len(bar_lengths)))
    bar_centers = bar_positions
    if ax is None:
        f, ax = plt.subplots()
    
    # Make the bar plot
    ax.bar(left=0, bottom=bar_centers, width=bar_lengths, height=.8, 
        align='center', xerr=bar_errs, capsize=0,
        ecolor='k', color=bar_colors, orientation='horizontal')
    
    # Hatch it
    if bar_hatches is not None:
        for p, hatch in zip(ax.patches, bar_hatches): p.set_hatch(hatch)
    
    # Plot squares on the bar tops
    ax.plot(bar_lengths, bar_centers, 'ks')
    
    # Labels
    ax.set_yticks(bar_centers)
    ax.set_yticklabels(bar_labels)
    
    return ax