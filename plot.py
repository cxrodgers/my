"""Wrapper functions with boilerplate code for making plots the way I like them
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import map
from builtins import range
from past.utils import old_div

import matplotlib
import matplotlib.patheffects as pe
import numpy as np, warnings
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats
from . import misc
import my
import pandas

def alpha_blend_with_mask(rgb0, rgb1, alpha0, mask0):
    """Alpha-blend two RGB images, masking out one image.
    
    rgb0 : first image, to be masked
        Must be 3-dimensional, and rgb0.shape[-1] must be 3 or 4
        If rgb0.shape[-1] == 4, the 4th channel will be dropped
    
    rgb1 : second image, wil not be masked
        Must be 3-dimensional, and rgb1.shape[-1] must be 3 or 4
        If rgb1.shape[-1] == 4, the 4th channel will be dropped
        Then, must have same shape as rgb0
    
    alpha0 : the alpha to apply to rgb0. (1 - alpha) will be applied to
    
    mask0 : True where to ignore rgb0
        Must have dimension 2 or 3
        If 2-dimensional, will be replicated along the channel dimension
        Then, must have same shape as rgb0
    
    Returns : array of same shape as rgb0 and rgb1
        Where mask0 is True, the result is the same as rgb1
        Where mask1 is False, the result is rgb0 * alpha0 + rgb1 * (1 - alpha0)
    """
    # Replicate mask along color channel if necessary
    if mask0.ndim == 2:
        mask0 = np.stack([mask0] * 3, axis=-1)
        
    # Check 3-dimensional
    assert mask0.ndim == 3
    assert rgb0.ndim == 3
    assert rgb1.ndim == 3

    # Drop alpha if present
    if rgb0.shape[-1] == 4:
        rgb0 = rgb0[:, :, :3]
    
    if rgb1.shape[-1] == 4:
        rgb1 = rgb1[:, :, :3]
    
    if mask0.shape[-1] == 4:
        mask0 = mask0[:, :, :3]
    
    # Error check
    assert rgb0.shape == rgb1.shape
    assert mask0.shape == rgb0.shape
    
    # Blend
    blended = alpha0 * rgb0 + (1 - alpha0) * rgb1
    
    # Flatten to apply mask
    blended_flat = blended.flatten()
    mask_flat = mask0.flatten()
    replace_with = rgb1.flatten()
    
    # Masked replace
    blended_flat[mask_flat] = replace_with[mask_flat]
    
    # Reshape to original
    replaced_blended = blended_flat.reshape(blended.shape)
    
    # Return
    return replaced_blended
    
def custom_RdBu_r():
    """Custom RdBu_r colormap with true white at center"""
    # Copied from matplotlib source: lib/matplotlib/_cm.py
    # And adjusted to go to true white at center
    _RdBu_data = (
        (0.40392156862745099,  0.0                ,  0.12156862745098039),
        (0.69803921568627447,  0.09411764705882353,  0.16862745098039217),
        (0.83921568627450982,  0.37647058823529411,  0.30196078431372547),
        (0.95686274509803926,  0.6470588235294118 ,  0.50980392156862742),
        (0.99215686274509807,  0.85882352941176465,  0.7803921568627451 ),
        (1,1,1),#(0.96862745098039216,  0.96862745098039216,  0.96862745098039216),
        (0.81960784313725488,  0.89803921568627454,  0.94117647058823528),
        (0.5725490196078431 ,  0.77254901960784317,  0.87058823529411766),
        (0.2627450980392157 ,  0.57647058823529407,  0.76470588235294112),
        (0.12941176470588237,  0.4                ,  0.67450980392156867),
        (0.0196078431372549 ,  0.18823529411764706,  0.38039215686274508)
        )

    # Copied from matplotlib source: lib/matplotlib/cm.py
    myrdbu = matplotlib.colors.LinearSegmentedColormap.from_list(
        'myrdbu', _RdBu_data[::-1], matplotlib.rcParams['image.lut'])

    # Return
    return myrdbu
    
def smooth_and_plot_versus_depth(
    data, 
    colname,
    ax=None,
    NS_sigma=40,
    RS_sigma=20,
    n_depth_bins=101,
    depth_min=0,
    depth_max=1600,
    datapoint_plot_kwargs=None,
    smoothed_plot_kwargs=None,
    plot_layer_boundaries=True,
    layer_boundaries_ylim=None,
    ):
    """Plot individual datapoints and smoothed versus depth.
    
    data : DataFrame
        Must have columns "Z_corrected", "NS", and `colname`, which become
        x- and y- coordinates.
    
    colname : string
        Name of column containing data
    
    ax : Axis, or None
        if None, creates ax
    
    NS_sigma, RS_sigma : float
        The standard deviation of the smoothing kernel to apply to each
    
    depth_min, depth_max, n_depth_bins : float, float, int
        The x-coordinates at which the smoothed results are evaluated
    
    datapoint_plot_kwargs : dict
        Plot kwargs for individual data points.
        Defaults: 
        'marker': 'o', 'ls': 'none', 'ms': 1.5, 'mew': 0, 'alpha': .25,
    
    smoothed_plot_kwargs : dict
        Plot kwargs for smoothed line.
        Defaults: 'lw': 1.5, 'path_effects': path_effects
    
    plot_layer_boundaries: bool
        If True, plot layer boundaries
    
    layer_boundaries_ylim : tuple of length 2, or None
        If not None, layer boundaries are plotted to these ylim
        If None, ax.get_ylim() is used after plotting everything else
    
    
    Returns: ax
    """
    ## Set up defaults
    # Bins at which to evaluate smoothed
    depth_bins = np.linspace(depth_min, depth_max, n_depth_bins)
    
    # datapoint_plot_kwargs
    default_datapoint_plot_kwargs = {
        'marker': 'o', 'ls': 'none', 'ms': 1, 'mew': 1, 
        'alpha': .3, 'mfc': 'none',
        }
    
    if datapoint_plot_kwargs is not None:
        default_datapoint_plot_kwargs.update(datapoint_plot_kwargs)
    
    use_datapoint_plot_kwargs = default_datapoint_plot_kwargs
    
    # smoothed_plot_kwargs
    path_effects = [pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]
    default_smoothed_plot_kwargs = {
        'lw': 1.5,
        'path_effects': path_effects,
        }
    
    if smoothed_plot_kwargs is not None:
        default_smoothed_plot_kwargs.update(smoothed_plot_kwargs)
    
    use_smoothed_plot_kwargs = default_smoothed_plot_kwargs        
    
    
    ## Plot versus depth
    if ax is None:
        f, ax = plt.subplots()

    # Iterate over NS
    for NS, sub_data in data.groupby('NS'):
        if NS:
            color = 'b'
            sigma = NS_sigma
        else:
            color = 'r'
            sigma = RS_sigma
        
        # Get the data to smooth
        to_smooth = sub_data.set_index('Z_corrected')[colname]
        
        # Smooth
        smoothed = my.misc.gaussian_sum_smooth_pandas(
            to_smooth, depth_bins, sigma=sigma)
        
        # Plot the individual data points
        ax.plot(
            to_smooth.index,
            to_smooth.values, 
            color=color,
            zorder=0,
            **use_datapoint_plot_kwargs,
            )
        
        # Plot the smoothed
        ax.plot(
            smoothed, color=color, 
            **use_smoothed_plot_kwargs)
    
    
    ## Pretty
    my.plot.despine(ax)

    ax.set_xticks((0, 500, 1000, 1500))
    ax.set_xlim((0, 1500))
    ax.set_xticklabels(('0.0', '0.5', '1.0', '1.5'))
    ax.set_xlabel('depth in cortex (mm)')
    
    
    ## Add layer boundaries
    if plot_layer_boundaries:
        # ylim for the boundaries
        if layer_boundaries_ylim is None:
            layer_boundaries_ylim = ax.get_ylim()
        
        # Layer boundaries
        layer_boundaries = [128, 419, 626, 1006, 1366]
        layer_names = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'L6b']
        
        # Centers of layers (for naming)
        layer_depth_bins = np.concatenate(
            [[-50], layer_boundaries, [1500]]).astype(np.float)
        layer_centers = (layer_depth_bins[:-1] + layer_depth_bins[1:]) / 2.0

        # Adjust position of L2/3 and L6 slightly
        layer_centers[1] = layer_centers[1] - 50
        layer_centers[2] = layer_centers[2] + 10
        layer_centers[3] = layer_centers[3] + 25
        layer_centers[-2] = layer_centers[-2] + 50
        

        # Plot each (but not top of L1 or bottom of L6)
        for lb in layer_boundaries[1:-1]:
            ax.plot(
                [lb, lb], layer_boundaries_ylim, 
                color='gray', lw=.8, zorder=-1)
    
        # Set the boundaries tight
        ax.set_ylim(layer_boundaries_ylim)
        
        # Warn
        if data[colname].max() > layer_boundaries_ylim[1]:
            print(
                "warning: max datapoint {} ".format(data[colname].max()) +
                "greater than layer_boundaries_ylim[1]")
        if data[colname].min() < layer_boundaries_ylim[0]:
            print(
                "warning: min datapoint {} ".format(data[colname].min()) +
                "less than layer_boundaries_ylim[0]")
        
        # Label the layer names
        # x in data, y in figure
        blended_transform = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.figure.transFigure)
        
        # Name each (but not L1 or L6b)
        zobj = zip(layer_names[1:-1], layer_centers[1:-1])
        for layer_name, layer_center in zobj:
            ax.text(
                layer_center, .98, layer_name, 
                ha='center', va='center', size=12, transform=blended_transform)

    
    ## Return ax
    return ax


def plot_by_depth_and_layer(df, column, combine_layer_5=True, aggregate='median',
    ax=None, ylim=None, agg_plot_kwargs=None, point_alpha=.5, point_ms=3,
    layer_label_offset=-.1, agg_plot_meth='rectangle'):
    """Plot values by depth and layer
    
    df : DataFrame
        Should have columns 'Z_corrected', 'layer', 'NS', and `column`
    column : name of column in `df` to plot
    combine_layer_5 : whether to combine 5a and 5b
    aggregate : None, 'mean', or 'median'
    ax : where to plot
    ylim : desired ylim (affects layer name position)
    agg_plot_kwargs : how to plot aggregated
    
    """
    # Set agg_plot_kwargs
    default_agg_plot_kwargs = {'marker': '_', 'ls': 'none', 'ms': 16, 
        'mew': 4, 'alpha': .5}
    if agg_plot_kwargs is not None:
        default_agg_plot_kwargs.update(agg_plot_kwargs)
    agg_plot_kwargs = default_agg_plot_kwargs
    
    # Layer boundaries
    layer_boundaries = [128, 419, 626, 1006, 1366]
    layer_names = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'L6b']
    layer_depth_bins = np.concatenate([[-50], layer_boundaries, [1500]]).astype(np.float)
    layer_centers = (layer_depth_bins[:-1] + layer_depth_bins[1:]) / 2.0
    
    # Make a copy
    df = df.copy()
    
    # Optionally combine layers 5a and 5b
    if combine_layer_5:
        # Combine layers 5a and 5b
        df['layer'] = df['layer'].astype(str)
        df.loc[df['layer'].isin(['5a', '5b']), 'layer'] = '5'

    # Optionally create figure
    if ax is None:
        f, ax = plt.subplots(figsize=(4.5, 3.5))

    # Plot datapoints for NS and RS separately
    NS_l = [False, True]
    for NS, sub_df in df.groupby('NS'):
        # Color by NS
        color = 'b' if NS else 'r'

        # Plot raw data
        ax.plot(
            sub_df.loc[:, 'Z_corrected'].values, 
            sub_df.loc[:, column].values,
            color=color, marker='o',  mfc='white',
            ls='none', alpha=point_alpha, ms=point_ms, clip_on=False,
        )

    # Keep track of this
    if ylim is None:
        ylim = ax.get_ylim()

    # Plot aggregates of NS and RS separately
    if aggregate is not None:
        for NS, sub_df in df.groupby('NS'):
            # Color by NS
            color = 'b' if NS else 'r' 
            
            # Aggregate over bins
            gobj = sub_df.groupby('layer')[column]
            counts_by_bin = gobj.size()
        
            # Aggregate
            if aggregate == 'mean':
                agg_by_bin = gobj.mean()
            elif aggregate == 'median':
                agg_by_bin = gobj.median()
            else:
                raise ValueError("unrecognized aggregated method: {}".format(aggregate))
            
            # Block out aggregates with too few data points
            agg_by_bin[counts_by_bin <= 3] = np.nan    
            
            # Reindex to ensure this matches layer_centers
            # TODO: Make this match the way it was aggregated
            agg_by_bin = agg_by_bin.reindex(['1', '2/3', '4', '5', '6', '6b'])
            assert len(agg_by_bin) == len(layer_centers)
            
            if agg_plot_meth == 'markers':
                # Plot aggregates as individual markers
                ax.plot(    
                    layer_centers,
                    agg_by_bin.values, 
                    color=color, 
                    **agg_plot_kwargs
                )
            
            elif agg_plot_meth == 'rectangle':
                # Plot aggregates as a rectangle
                for n_layer, layer in enumerate(['2/3', '4', '5', '6']):
                    lo_depth = layer_depth_bins[n_layer + 1]
                    hi_depth = layer_depth_bins[n_layer + 2]
                    value = agg_by_bin.loc[layer]
                    
                    #~ ax.plot([lo_depth, hi_depth], [value, value], 
                        #~ color='k', ls='-', lw=2.5)
                    #~ ax.plot([lo_depth, hi_depth], [value, value], 
                        #~ color=color, ls='--', lw=2.5)
                    
                    # zorder brings the patch on top of the datapoints
                    patch = plt.Rectangle(
                        (lo_depth + .1 * (hi_depth - lo_depth), value), 
                        width=((hi_depth-lo_depth) * .8), 
                        height=(.03 * np.diff(ylim)), 
                        ec='k', fc=color, alpha=.5, lw=1.5, zorder=20)

                    ax.add_patch(patch)

    # Plot layer boundaries, skipping L1 and L6b
    for lb in layer_boundaries[1:-1]:
        ax.plot([lb, lb], [ylim[0], ylim[1]], color='gray', ls='-', lw=1)

    # Name the layers
    text_ypos = ylim[1] + layer_label_offset * (ylim[1] - ylim[0])
    for layer_name, layer_center in zip(layer_names, layer_centers):
        if layer_name in ['L1', 'L6b']:
            continue
        ax.text(layer_center, text_ypos, layer_name[1:], ha='center', va='bottom', 
            color='k')
    
    # Reset the ylim
    ax.set_ylim(ylim)

    # xticks
    ax.set_xticks((200, 600, 1000, 1400))
    ax.set_xticklabels([])
    ax.set_xlim((100, 1500))
    my.plot.despine(ax)
    ax.set_xlabel('depth in cortex')
    
    return ax
    
def connected_pairs(v1, v2, p=None, signif=None, shapes=None, colors=None, 
    labels=None, ax=None):
    """Plot columns of (v1, v2) as connected pairs"""
    import my.stats
    if ax is None:
        f, ax = plt.subplots()
    
    # Arrayify 
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if signif is None:
        signif = np.zeros_like(v1)
    else:
        signif = np.asarray(signif)
    
    # Defaults
    if shapes is None:
        shapes = ['o'] * v1.shape[0]
    if colors is None:
        colors = ['k'] * v1.shape[0]
    if labels is None:
        labels = ['' * v1.shape[1]]
    
    # Store location of each pair
    xvals = []
    xvalcenters = []
    
    # Iterate over columns
    for n, (col1, col2, signifcol, label) in enumerate(zip(v1.T, v2.T, signif.T, labels)):
        # Where to plot this pair
        x1 = n * 2
        x2 = n * 2 + 1
        xvals += [x1, x2]
        xvalcenters.append(np.mean([x1, x2]))
        
        # Iterate over specific pairs
        for val1, val2, sigval, shape, color in zip(col1, col2, signifcol, shapes, colors):
            lw = 2 if sigval else 0.5
            ax.plot([x1, x2], [val1, val2], marker=shape, color=color, 
                ls='-', mec=color, mfc='none', lw=lw)
        
        # Plot the median
        median1 = np.median(col1[~np.isnan(col1)])
        median2 = np.median(col2[~np.isnan(col2)])
        ax.plot([x1, x2], [median1, median2], marker='o', color='k', ls='-',
            mec=color, mfc='none', lw=4)
        
        # Sigtest on pop
        utest_res = my.stats.r_utest(col1[~np.isnan(col1)], col2[~np.isnan(col2)],
            paired='TRUE', fix_float=1e6)
        if utest_res['p'] < 0.05:
            ax.text(np.mean([x1, x2]), 1.0, '*', va='top', ha='center')
    
    # Label center of each pair
    ax.set_xlim([xvals[0]-1, xvals[-1] + 1])
    if labels:
        ax.set_xticks(xvalcenters)
        ax.set_xticklabels(labels)
    
    return ax, xvals

def radar_by_stim(evoked_resp, ax=None, label_stim=True):
    """Given a df of spikes by stim, plot radar
    
    evoked_resp should have arrays of counts indexed by all the stimulus
    names
    """
    from ns5_process import LBPB
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 3), subplot_kw={'polar': True})

    # Heights of the bars
    evoked_resp = evoked_resp.ix[LBPB.mixed_stimnames]
    barmeans = evoked_resp.apply(np.mean)
    barstderrs = evoked_resp.apply(misc.sem)
    
    # Set up the radar
    radar_dists = [[barmeans[sname+block] 
        for sname in ['ri_hi', 'le_hi', 'le_lo', 'ri_lo']] 
        for block in ['_lc', '_pc']]
    
    # make it circular
    circle_meansLB = np.array(radar_dists[0] + [radar_dists[0][0]])
    circle_meansPB = np.array(radar_dists[1] + [radar_dists[1][0]])
    circle_errsLB = np.array([barstderrs[sname+'_lc'] for sname in 
        ['ri_hi', 'le_hi', 'le_lo', 'ri_lo', 'ri_hi']])
    circle_errsPB = np.array([barstderrs[sname+'_pc'] for sname in 
        ['ri_hi', 'le_hi', 'le_lo', 'ri_lo', 'ri_hi']])
    
    # x-values (really theta values)
    xts = np.array([45, 135, 225, 315, 405])*np.pi/180.0
    
    # Plot LB means and errs
    #ax.errorbar(xts, circle_meansLB, circle_errsLB, color='b')
    ax.plot(xts, circle_meansLB, color='b')
    ax.fill_between(x=xts, y1=circle_meansLB-circle_errsLB,
        y2=circle_meansLB+circle_errsLB, color='b', alpha=.5)
    
    # Plot PB means and errs
    ax.plot(xts, circle_meansPB, color='r')
    ax.fill_between(x=xts, y1=circle_meansPB-circle_errsPB,
        y2=circle_meansPB+circle_errsPB, color='r', alpha=.5)
    
    # Tick labels
    xtls = ['right\nhigh', 'left\nhigh', 'left\nlow', 'right\nlow']        
    ax.set_xticks(xts)
    ax.set_xticklabels([]) # if xtls, will overlap
    ax.set_yticks(ax.get_ylim()[1:])
    ax.set_yticks([])
        
    # manual tick
    if label_stim:
        for xt, xtl in zip(xts, xtls):
            ax.text(xt, ax.get_ylim()[1]*1.25, xtl, size='large', 
                ha='center', va='center')            
    
    # pretty and save
    #f.tight_layout()
    return ax
    

def despine(ax, detick=True, which_ticks='both', which=('right', 'top')):
    """Remove the top and right axes from the plot
    
    which_ticks : can be 'major', 'minor', or 'both
    """
    for w in which:
        ax.spines[w].set_visible(False)
        if detick:
            ax.tick_params(which=which_ticks, **{w:False})
    return ax

def font_embed():
    """Produce files that can be usefully imported into AI"""
    # For PDF imports:
    # Not sure what this does
    matplotlib.rcParams['ps.useafm'] = True
    
    # Makes it so that the text is editable
    matplotlib.rcParams['pdf.fonttype'] = 42
    
    # For SVG imports:
    # AI can edit the text but can't import the font itself
    #matplotlib.rcParams['svg.fonttype'] = 'svgfont'
    
    # seems to work better
    matplotlib.rcParams['svg.fonttype'] = 'none'

def manuscript_defaults():
    """For putting into a word document.
    
    Typical figure is approx 3"x3" panels. Apply a 50% scaling.
    I think these defaults should be 14pt, actually.
    """
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    matplotlib.rcParams['font.size'] = 14 # ax.text objects
    matplotlib.rcParams['legend.fontsize'] = 14

def poster_defaults():
    """For a poster
    
    Title: 80pt
    Section headers: 60pt
    Body text: 40pt
    Axis labels, tick marks, subplot titles: 32pt
    
    Typical panel size: 6"
    So it's easiest to just use manuscript_defaults() and double
    the size.
    """
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    matplotlib.rcParams['font.size'] = 14 # ax.text objects
    matplotlib.rcParams['legend.fontsize'] = 14

def presentation_defaults():
    """For importing into presentation.
    
    Typical figure is 11" wide and 7" tall. No scaling should be necessary.
    Typically presentation figures have more whitespace and fewer panels
    than manuscript figures.
    
    Actually I think the font size should not be below 18, unless really
    necessary.
    """
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['axes.labelsize'] = 18
    matplotlib.rcParams['axes.titlesize'] = 18
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['font.size'] = 18 # ax.text objects
    matplotlib.rcParams['legend.fontsize'] = 18

def figure_1x1_small():
    """Smaller f, ax for single panel with a nearly square axis

    """
    f, ax = plt.subplots(figsize=(2.2, 2))
    
    # left = .3 is for the case of yticklabels with two signif digits
    f.subplots_adjust(bottom=.28, left=.3, right=.95, top=.95)
    return f, ax
    
def figure_1x1_square():
    """Standard size f, ax for single panel with a square axis
    
    Room for xlabel, ylabel, and title in 16pt font
    """
    f, ax = plt.subplots(figsize=(3, 3))
    f.subplots_adjust(bottom=.23, left=.26, right=.9, top=.87)
    return f, ax

def figure_1x1_standard():
    """Standard size f, ax for single panel with a slightly rectangular axis
    
    Room for xlabel, ylabel, and title in 16pt font
    """
    f, ax = plt.subplots(figsize=(3, 2.5))
    f.subplots_adjust(bottom=.24, left=.26, right=.93, top=.89)
    return f, ax

def figure_1x2_standard(**kwargs):
    """Standard size f, ax for single panel with a slightly rectangular axis
    
    Room for xlabel, ylabel, and title in 16pt font
    """
    f, axa = plt.subplots(1, 2, figsize=(6, 2.5), **kwargs)
    f.subplots_adjust(left=.15, right=.9, wspace=.2, bottom=.22, top=.85)

    return f, axa

def figure_1x2_small(**kwargs):
    f, axa = plt.subplots(1, 2, figsize=(4, 2), **kwargs)
    f.subplots_adjust(left=.2, right=.975, wspace=.3, bottom=.225, top=.8)

    return f, axa
    
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

def crucifix(x, y, xerr=None, yerr=None, relative_CIs=False, p=None, 
    ax=None, factor=None, below=None, above=None, null=None,
    data_range=None, axtype=None, zero_substitute=1e-6,
    suppress_null_error_bars=False):
    """Crucifix plot y vs x around the unity line
    
    x, y : array-like, length N, paired data
    xerr, yerr : array-like, Nx2, confidence intervals around x and y
    relative_CIs : if True, then add x to xerr (and ditto yerr)
    p : array-like, length N, p-values for each point
    ax : graphical object
    factor : multiply x, y, and errors by this value
    below : dict of point specs for points significantly below the line
    above : dict of point specs for points significantly above the line
    null : dict of point specs for points nonsignificant
    data_range : re-adjust the data limits to this
    axtype : if 'symlog' then set axes to symlog
    """
    # Set up point specs
    if below is None:
        below = {'color': 'b', 'marker': '.', 'ls': '-', 'alpha': 1.0,
            'mec': 'b', 'mfc': 'b'}
    if above is None:
        above = {'color': 'r', 'marker': '.', 'ls': '-', 'alpha': 1.0,
            'mec': 'r', 'mfc': 'r'}
    if null is None:
        null = {'color': 'gray', 'marker': '.', 'ls': '-', 'alpha': 0.5,
            'mec': 'gray', 'mfc': 'gray'}
    
    # Defaults for data range
    if data_range is None:
        data_range = [None, None]
    else:
        data_range = list(data_range)
    
    # Convert to array and optionally multiply
    if factor is None:
        factor = 1
    x = np.asarray(x) * factor
    y = np.asarray(y) * factor

    # p-values
    if p is not None: 
        p = np.asarray(p)
    
    # Same with errors but optionally also reshape and recenter
    if xerr is not None: 
        xerr = np.asarray(xerr) * factor
        if xerr.ndim == 1:
            xerr = np.array([-xerr, xerr]).T
        if relative_CIs:
            xerr += x[:, None]
    if yerr is not None: 
        yerr = np.asarray(yerr) * factor
        if yerr.ndim == 1:
            yerr = np.array([-yerr, yerr]).T
        if relative_CIs:
            yerr += y[:, None]
    
    # Create figure handles
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    # Plot each point
    min_value, max_value = [], []
    for n, (xval, yval) in enumerate(zip(x, y)):
        # Get p-value and error bars for this point
        pval = 1.0 if p is None else p[n]
        xerrval = xerr[n] if xerr is not None else None
        yerrval = yerr[n] if yerr is not None else None
        
        # Replace neginfs
        if xerrval is not None:
            xerrval[xerrval == 0] = zero_substitute
        if yerrval is not None:
            yerrval[yerrval == 0] = zero_substitute
        
        #~ if xval < .32:
            #~ 1/0
        
        # What color
        if pval < .05 and yval < xval:
            pkwargs = below
        elif pval < .05 and yval > xval:
            pkwargs = above
        else:
            pkwargs = null
        lkwargs = pkwargs.copy()
        lkwargs.pop('marker')
        
        # Now actually plot the point
        ax.plot([xval], [yval], **pkwargs)
        
        # plot error bars, keep track of data range
        if xerrval is not None and not (suppress_null_error_bars and pkwargs is null):
            ax.plot(xerrval, [yval, yval], **lkwargs)
            max_value += list(xerrval)
        else:
            max_value.append(xval)
        
        # same for y
        if yerrval is not None and not (suppress_null_error_bars and pkwargs is null):
            ax.plot([xval, xval], yerrval, **lkwargs)
            max_value += list(yerrval)
        else:
            max_value.append(xval)        

    # Plot the unity line
    if data_range[0] is None:
        data_range[0] = np.min(max_value)
    if data_range[1] is None:
        data_range[1] = np.max(max_value)
    ax.plot(data_range, data_range, 'k:')
    ax.set_xlim(data_range)
    ax.set_ylim(data_range)
    
    # symlog
    if axtype:
        ax.set_xscale(axtype)
        ax.set_yscale(axtype)
    
    ax.axis('scaled')
    
    
    return ax

def scatter_with_trend(x, y, xname='X', yname='Y', ax=None, 
    legend_font_size='medium', **kwargs):
    """Scatter plot `y` vs `x`, also linear regression line
    
    Kwargs sent to the point plotting
    """
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    if 'ls' not in kwargs:
        kwargs['ls'] = ''
    if 'color' not in kwargs:
        kwargs['color'] = 'g'
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    dropna = np.isnan(x) | np.isnan(y)
    x = x[~dropna]
    y = y[~dropna]
    
    if ax is None:    
        f = plt.figure()
        ax = f.add_subplot(111)
    ax.plot(x, y, **kwargs)

    m, b, rval, pval, stderr = \
        scipy.stats.stats.linregress(x.flatten(), y.flatten())
    
    trend_line_label = 'r=%0.3f p=%0.3f' % (rval, pval)
    ax.plot([x.min(), x.max()], m * np.array([x.min(), x.max()]) + b, 'k:',
        label=trend_line_label)
    ax.legend(loc='best', prop={'size':legend_font_size})
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    return ax

def vert_bar(bar_lengths, bar_labels=None, bar_positions=None, ax=None,
    bar_errs=None, bar_colors=None, bar_hatches=None, tick_labels_rotation=90,
    plot_bar_ends='ks', bar_width=.8, mpl_ebar=False,
    yerr_is_absolute=True):
    """Vertical bar plot with nicer defaults
    
    bar_lengths : heights of the bars, length N
    bar_labels : text labels
    bar_positions : x coordinates of the bar centers. Default is range(N)
    ax : axis to plot in
    bar_errs : error bars. Will be cast to array
        If 1d, then these are drawn +/-
        If 2d, then (UNLIKE MATPLOTLIB) they are interpreted as the exact
        locations of the endpoints. Transposed as necessary. If mpl_ebar=True, 
        then it is passed directly to `errorbar`, and it needs to be 2xN and
        the bars are drawn at -row0 and +row1.
    bar_colors : colors of bars. If longer than N, then the first N are taken
    bar_hatches : set the hatches like this. length N
    plot_bar_ends : if not None, then this is plotted at the tops of the bars
    bar_width : passed as width to ax.bar
    mpl_ebar : controls behavior of errorbars
    yerr_is_absolute : if not mpl_ebar, and you are independently specifying
        the locations of each end exactly, set this to True
        Does nothing if yerr is 1d
    """
    # Default bar positions
    if bar_positions is None:
        bar_positions = list(range(len(bar_lengths)))
    bar_centers = bar_positions
    
    # Arrayify bar lengths
    bar_lengths = np.asarray(bar_lengths)
    N = len(bar_lengths)
    
    # Default bar colors
    if bar_colors is not None:
        bar_colors = np.asarray(bar_colors)
        if len(bar_colors) > N:
            bar_color = bar_colors[:N]
    
    # Deal with errorbars (if specified, and not mpl_ebar behavior)
    if bar_errs is not None and not mpl_ebar:
        bar_errs = np.asarray(bar_errs)
        
        # Transpose as necessary
        if bar_errs.ndim == 2 and bar_errs.shape[0] != 2:
            if bar_errs.shape[1] == 2:
                bar_errs = bar_errs.T
            else:
                raise ValueError("weird shape for bar_errs: %r" % bar_errs)
        
        if bar_errs.ndim == 2 and yerr_is_absolute:
            # Put into MPL syntax: -row0, +row1
            assert bar_errs.shape[1] == N
            bar_errs = np.array([
                bar_lengths - bar_errs[0],
                bar_errs[1] - bar_lengths])
    
    # Create axis objects
    if ax is None:
        f, ax = plt.subplots()
    
    # Make the bar plot
    ax.bar(left=bar_centers, bottom=0, width=bar_width, height=bar_lengths, 
        align='center', yerr=bar_errs, capsize=0,
        ecolor='k', color=bar_colors, orientation='vertical')
    
    # Hatch it
    if bar_hatches is not None:
        for p, hatch in zip(ax.patches, bar_hatches): p.set_hatch(hatch)
    
    # Plot squares on the bar tops
    if plot_bar_ends:
        ax.plot(bar_centers, bar_lengths, plot_bar_ends)
    
    # Labels
    ax.set_xticks(bar_centers)
    ax.set_xlim(bar_centers[0] - bar_width, bar_centers[-1] + bar_width)
    if bar_labels:
        ax.set_xticklabels(bar_labels, rotation=tick_labels_rotation)
    
    return ax

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

def auto_subplot(n, return_fig=True, squeeze=False, **kwargs):
    """Return nx and ny for n subplots total"""
    nx = int(np.floor(np.sqrt(n)))
    ny = int(np.ceil(old_div(n, float(nx))))
    
    if return_fig:
        return plt.subplots(nx, ny, squeeze=squeeze, **kwargs)
    else:
        return nx, ny

def imshow(C, x=None, y=None, ax=None, 
    extent=None, xd_range=None, yd_range=None,
    cmap=plt.cm.RdBu_r, origin='upper', interpolation='nearest', aspect='auto', 
    axis_call='tight', clim=None, center_clim=False, 
    skip_coerce=False, **kwargs):
    """Wrapper around imshow with better defaults.
    
    Plots "right-side up" with the first pixel C[0, 0] in the upper left,
    not the lower left. So it's like an image or a matrix, not like
    a graph. This done by setting the `origin` to 'upper', and by 
    appropriately altering `extent` to account for this flip.
    
    C must be regularly-spaced. See this example for how to handle irregular:
    http://stackoverflow.com/questions/14120222/matplotlib-imshow-with-irregular-spaced-data-points
    
    C - Two-dimensional array of data
        If C has shape (m, n), then the image produced will have m rows
        and n columns.
    x - array of values corresonding to the x-coordinates of the columns
        Only use this if you want numerical values on the columns.
        Because the spacing must be regular, the way this is handled is by
        setting `xd_range` to the first and last values in `x`
    y - Like x but for rows.
    ax - Axes object
        If None, a new one is created.
    axis_call - string
        ax.axis is called with this. The default `tight` fits the data
        but does not constrain the pixels to be square.
    extent - tuple, or None
        (left, right, bottom, top) axis range
        Note that `bottom` and `top` are really the bottom and top labels.
        So, generally you will want to provide this as (xmin, xmax, ymax, ymin),
        or just provide xd_range and yd_range and this function will handle
        the swap.
    xd_range - 2-tuple, or None
        If you want the x-coordinates to have numerical labels, use this.
        Specify the value of the first and last column.
        If None, then 0-based integer indexing is assumed.
        NB: half a pixel is subtracted from the first and added to the last
        to calculate the `extent`.
    yd_range - 2-tuple, or None
        Like xd_range but for rows.
        Always provide this as (y-coordinate of first row of C, y-coordinate
        of last row of C). It will be flipped as necessary to match the data.
    cmap, origin, interpolation, aspect
        just like plt.imshow, but different defaults
    clim : Tuple, or None
        Color limits to apply to image
        See also `harmonize_clim`
    
    Returns: Image object
    """
    # Coerce data to array
    if not skip_coerce:
        C = np.asarray(C)
    
    # Set up axis if necessary
    if ax is None:
        f, ax = plt.subplots()
    
    # Data range
    if extent is None:
        # Specify the data range with 0-based indexing if necessary
        if xd_range is None:
            if x is None:
                xd_range = (0, C.shape[1] - 1)
            else:
                if len(x) != C.shape[1]:
                    warnings.warn("x-labels do not match data size")
                xd_range = (x[0], x[-1])
        if yd_range is None:
            if y is None:
                yd_range = (0, C.shape[0] - 1)
            else:
                if len(y) != C.shape[0]:
                    warnings.warn("y-labels do not match data size")                
                yd_range = (y[0], y[-1])
        
        # Calculate extent from data range by adding (subtracting) half a pixel
        try:
            xwidth = old_div((xd_range[1] - xd_range[0]), (C.shape[1] - 1))
        except ZeroDivisionError:
            xwidth = 1.
        try:
            ywidth = old_div((yd_range[1] - yd_range[0]), (C.shape[0] - 1))
        except ZeroDivisionError:
            ywidth=1.
        extent = (
            xd_range[0] - old_div(xwidth,2.), xd_range[1] + old_div(xwidth,2.),
            yd_range[0] - old_div(ywidth,2.), yd_range[1] + old_div(ywidth,2.))

        # Optionally invert the yd_range
        # Because we specify the `extent` manually, we also need to correct
        # it for origin == 'upper'
        if origin == 'upper':
            extent = extent[0], extent[1], extent[3], extent[2]
    
    # Actual call to imshow
    im = ax.imshow(C, interpolation=interpolation, origin=origin,
        extent=extent, aspect=aspect, cmap=cmap, **kwargs)
    
    # Fix up the axes
    ax.axis(axis_call)
    
    # Deal with color limits
    if clim is not None:
        im.set_clim(clim)
    
    return im

def colorbar(ax=None, fig=None, new_wspace=.4, **kwargs):
    """Insert colorbar into axis or every axis in a figure."""
    # Separate behavior based on fig
    if fig:
        if new_wspace:
            fig.subplots_adjust(wspace=new_wspace)
        
        # Colorbar for all contained axes
        for ax in fig.axes:
            if ax.images and len(ax.images) > 0:
                c = fig.colorbar(ax.images[0], ax=ax, **kwargs)
    
    else:
        # Colorbar just for ax
        fig = ax.figure
        if ax.images and len(ax.images) > 0:    
            c = fig.colorbar(ax.images[0], ax=ax, **kwargs)
    
    return c

def harmonize_clim_in_subplots(fig=None, axa=None, clim=(None, None), 
    center_clim=False, trim=1):
    """Set clim to be the same in all subplots in figur
    
    f : Figure to grab all axes from, or None
    axa : the list of subplots (if f is None)
    clim : tuple of desired c-limits. If either or both values are
        unspecified, they are derived from the data.
    center_clim : if True, the mean of the new clim is always zero
        May overrule specified `clim`
    trim : does nothing if 1 or None
        otherwise, sets the clim to truncate extreme values
        for example, if .99, uses the 1% and 99% values of the data
    """
    # Which axes to operate on
    if axa is None:
        axa = fig.get_axes()
    axa = np.asarray(axa)

    # Two ways of getting new clim
    if trim is None or trim == 1:
        # Get all the clim
        all_clim = []        
        for ax in axa.flatten():
            for im in ax.get_images():
                all_clim.append(np.asarray(im.get_clim()))
        
        # Find covering clim and optionally center
        all_clim_a = np.array(all_clim)
        new_clim = [np.min(all_clim_a[:, 0]), np.max(all_clim_a[:, 1])]
    else:
        # Trim to specified prctile of the image data
        data_l = []
        for ax in axa.flatten():
            for im in ax.get_images():
                data_l.append(np.asarray(im.get_array()).flatten())
        data_a = np.concatenate(data_l)
        
        # New clim
        new_clim = list(np.percentile(data_a, (100.*(1-trim), 100.*trim)))
    
    # Take into account specified clim
    try:
        if clim[0] is not None:
            new_clim[0] = clim[0]
        if clim[1] is not None:
            new_clim[1] = clim[1]
    except IndexError:
        print("warning: problem with provided clim")
    
    # Optionally center
    if center_clim:
        new_clim = np.max(np.abs(new_clim)) * np.array([-1, 1])
    
    # Set to new value
    for ax in axa.flatten():
        for im in ax.get_images():
            im.set_clim(new_clim)
    
    return new_clim

def generate_colorbar(n_colors, mapname='jet', rounding=100, start=0., stop=1.):
    """Generate N evenly spaced colors from start to stop in map"""
    color_idxs = my.rint(rounding * np.linspace(start, stop, n_colors))[::-1]
    colors = plt.cm.get_cmap('jet', rounding)(color_idxs)
    return colors

def pie(n_list, labels, ax=None, autopct=None, colors=None):
    """Make a pie chart
    
    n_list : list of integers, size of each category
    labels : list of strings, label for each category
    colors : list of strings convertable to colors
    autopct : function taking a percentage and converting it to a label
        Default converts it to "N / N_total"
    
    """
    # How to create the percentage strings
    n_total = np.sum(n_list)
    def percent_to_fraction(pct):
        n = int(np.rint(pct / 100. * n_total))
        return '%d/%d' % (n, n_total)
    if autopct is None:
        autopct = percent_to_fraction

    # Create the figure
    if ax is None:
        f, ax  = plt.subplots()
        f.subplots_adjust(left=.23, right=.81)

    # Plot it
    patches, texts, pct_texts = ax.pie(
        n_list, colors=colors,
        labels=labels, 
        explode=[.1]*len(n_list),
        autopct=percent_to_fraction)

    #for t in texts: t.set_horizontalalignment('center')
    for t in pct_texts: 
        plt.setp(t, 'color', 'w', 'fontweight', 'bold')
    
    ax.axis('equal')
    return ax


def hist_p(data, p, bins=20, thresh=.05, ax=None, **hist_kwargs):
    """Make a histogram with significant entries colored differently"""
    if ax is None:
        f, ax = plt.subplots()
    
    if np.sum(p > thresh) == 0:
        # All nonsig
        ax.hist(data[p<=thresh], bins=bins, histtype='barstacked', color='r',
            **hist_kwargs)    
    elif np.sum(p < thresh) == 0:
        # All sig
        ax.hist(data[p>thresh], bins=bins, histtype='barstacked', color='k',
            **hist_kwargs)            
    else:
        # Mixture
        ax.hist([data[p>thresh], data[p<=thresh]], bins=bins, 
            histtype='barstacked', color=['k', 'r'], rwidth=1.0, **hist_kwargs)    
    return ax


def errorbar_data(data=None, x=None, ax=None, errorbar=True, axis=0, 
    fill_between=False, fb_kwargs=None, eb_kwargs=None, error_fn=misc.sem,
    **kwargs):
    """Plots mean and SEM for a matrix `data`
    
    data : 1d or 2d
    axis : if 0, then replicates in `data` are along rows
    x : corresponding x values for points in data
    ax : where to plot
    errorbar : if True and if 2d, will plot SEM
        The format of the error bars depends on fill_between
    eb_kwargs : kwargs passed to errorbar
    fill_between: whether to plots SEM as bars or as trace thickness
    fb_kwargs : kwargs passed to fill_between
    error_fn : how to calculate error bars
    
    Other kwargs are passed to `plot`
    Returns the axis object
    """
    if ax is None:
        f, ax = plt.subplots(1, 1)
    
    # plotting defaults
    if fb_kwargs is None:
        fb_kwargs = {}
    if eb_kwargs is None:
        eb_kwargs = {}
    if 'capsize' not in eb_kwargs:
        eb_kwargs['capsize'] = 0
    if 'lw' not in fb_kwargs:
        fb_kwargs['lw'] = 0
    if 'alpha' not in fb_kwargs:
        fb_kwargs['alpha'] = .5
    if 'color' in kwargs and 'color' not in fb_kwargs:
        fb_kwargs['color'] = kwargs['color']
    if 'color' in kwargs and 'color' not in eb_kwargs:
        eb_kwargs['color'] = kwargs['color']


    # Put data into 2d, or single trace
    data = np.asarray(data)
    if np.min(data.shape) == 1:
        data = data.flatten()
    if data.ndim == 1:
        #single trace
        single_trace = True
        errorbar = False        
        if x is None:
            x = list(range(len(data)))
    else:
        single_trace = False        
        if x is None:
            x = list(range(len(np.mean(data, axis=axis))))
    
    # plot
    if single_trace:
        ax.plot(x, data, **kwargs)
    else:
        if errorbar:
            y = np.mean(data, axis=axis)
            yerr = error_fn(data, axis=axis)
            if fill_between:
                ax.plot(x, y, **kwargs)
                ax.fill_between(x, y1=y-yerr, y2=y+yerr, **fb_kwargs)
            else:
                ax.errorbar(x=x, y=y, yerr=yerr, **eb_kwargs)
        else:
            ax.plot(np.mean(data, axis=axis), **kwargs)
    
    return ax


## Grouped bar plot stuff
def index2plot_kwargs__shape_task(ser):
    """Given Series, return plot_kwargs as dict"""
    
    if 'rewside' in ser:
        if ser['rewside'] == 'left':
            color = 'b'
        elif ser['rewside'] == 'right':
            color = 'r'
        else:
            raise ValueError("unknown rewside")
    else:
        color = 'k'
    
    if 'servo_pos' in ser:
        if ser['servo_pos'] == 1670:
            alpha = .3
        elif ser['servo_pos'] == 1760:
            alpha = .5
        elif ser['servo_pos'] == 1850:
            alpha = .7
        else:
            raise ValueError("unknown servo_pos")
    else:
        alpha = 1
    
    if 'outcome' in ser:
        if ser['outcome'] == 'hit':
            ec = 'none'
            fc = color
        elif ser['outcome'] == 'error':
            fc = 'w'
            ec = color
        else:
            raise ValueError("unknown outcome")
    else:
        # If no outcome specified, use filled bars
        ec = 'none'
        fc = color

    res = {'ec': ec, 'alpha': alpha, 'fc': fc}
    return res

def index2label(ser):
    """Given series, return xtick label"""
    return ' '.join(map(str, ser.values))

def index2label__shape_task(ser):
    """Given series, return xtick label"""
    if 'rewside' in ser:
        if ser['rewside'] == 'left':
            stim = 'CC'
        elif ser['rewside'] == 'right':
            stim = 'CV'
    else:
        stim = ''
    
    if 'servo_pos' in ser:
        if ser['servo_pos'] == 1670:
            servo_pos = 'far'
        elif ser['servo_pos'] == 1760:
            servo_pos = 'med'
        elif ser['servo_pos'] == 1850:
            servo_pos = 'close'
        else:
            raise ValueError("unknown servo_pos")
    else:
        servo_pos = ''
    
    if 'outcome' in ser:
        if ser['outcome'] == 'hit':
            outcome = 'hit'
        elif ser['outcome'] == 'error':
            outcome = 'error'
        else:
            raise ValueError("unknown outcome")
    else:
        outcome = ''

    return ' '.join([servo_pos, stim, outcome])


def group_index2group_label__rewside2shape(group_index):
    if group_index == 'left':
        group_label = 'concave'
    elif group_index == 'right':
        group_label = 'convex'
    else:
        group_label = None

    return group_label

def grouped_bar_plot(df, 
    index2plot_kwargs, 
    index2label=None, 
    group_index2group_label=None, 
    yerrlo=None, yerrhi=None, 
    ax=None, 
    xtls_kwargs=None, group_name_kwargs=None,
    datapoint_plot_kwargs=None,
    group_name_y_offset=None,
    group_name_fig_ypos=.1,
    plot_error_bars_instead_of_points=False,
    elinewidth=.75,
    unclip_error_bars=True,
    ):
    """Plot groups of bars
    
    df : DataFrame
        The columns are considered replicates to aggregate over
        The levels of the index determine the grouping
        The columns will be plotted in exactly the order they are provided
    
    index2plot_kwargs : function
        Takes one of the entries in df.index and returns plot_kwargs
        Currently accepts only 'alpha', 'ec', 'fc', and 'lw'
        Example: index2plot_kwargs__shape_task
    
    index2label : function or None
        Used to create the labels for the xticks.
        Applied to `df.index` if `df.index.nlevels == 1`, else the index 
        after dropping the top level (group name).
        Example: index2label__shape_task        
    
    group_index2group_label : function taking a string, or None
        Used to label the groups
        By default, the actual values in the index (the "levels") are used
        Example: group_index2group_label__rewside2shape
    
    group_name_fig_ypos : y-position of the group labels, in figure coordinates
    
    yerrlo, yerrhi : DataFrame or None
        The index must match df.index exactly
    
    plot_error_bars_instead_of_points : bool
        If True, plot standard error bars instead of raw datapoints
    
    Returns: ax, bar_container
    """
    ## Create figure handles if needed
    if ax is None:
        f, ax = plt.subplots()
    
    
    ## Default values
    if xtls_kwargs is None:
        xtls_kwargs = {}
    
    if group_name_kwargs is None:
        group_name_kwargs = {}
    
    
    ## Deal with error bars versus points
    if plot_error_bars_instead_of_points:
        assert df.ndim == 2
        yerrlo = df.mean(1) - df.sem(1)
        yerrhi = df.mean(1) + df.sem(1)
        df = df.mean(1)
    
    
    ## Remove levels
    df = df.copy()
    try:
        df.index = df.index.remove_unused_levels()
    except AttributeError:
        pass
    try:
        df.columns = df.columns.remove_unused_levels()
    except AttributeError:
        pass
    
    
    ## Error check that indices of df and yerr are aligned
    # Below they are just taken directly as arrays
    if yerrlo is not None:
        assert (df.index == yerrlo.index).all()
    
    if yerrhi is not None:
        assert (df.index == yerrhi.index).all()


    ## Datapoint plot kwargs
    default_datapoint_plot_kwargs = {
        'marker': 'o', 'color': 'k', 'mfc': 'none', 'ls': 'none', 
        'clip_on': False}
    if datapoint_plot_kwargs is not None:
        default_datapoint_plot_kwargs.update(datapoint_plot_kwargs)
    datapoint_plot_kwargs = default_datapoint_plot_kwargs
    
    
    ## Generate xts and xt_group_centers by iterating over groups
    offset = 0
    xts_l = []
    xt_group_centers = []
    
    # Depends on whether the index is a MultiIndex
    if df.index.nlevels > 1:
        ## Multi-level, group on first level
        # Take the order of the group names from the order they occur
        # in level 0, which is not necessarily sorted, and not necessarily
        # ordered like df.index.levels[0]
        group_names = df.index.get_level_values(0).drop_duplicates()
        
        # Error check that the groups are contiguous
        new_df = pandas.concat(
            [df.xs(group_name, level=0, drop_level=False) 
            for group_name in group_names]
            )
        try:
            assert 999 not in new_df.values
            assert 999 not in df.values
            assert (new_df.fillna(999) == df.fillna(999)).all().all()
            assert (new_df.isnull() == df.isnull()).all().all()
        except ValueError:
            raise ValueError('df must be contiguous on the first level')
        
        # Count the size of each group, and generate xticks within
        offset = 0
        for group_idx in df.index.levels[0]:
            group_len = len(df.loc[group_idx])
            to_append = offset + np.arange(group_len, dtype=np.int)
            xt_group_centers.append(to_append.mean())
            xts_l.append(to_append)
            offset += group_len + 1
        
        xts = np.concatenate(xts_l)
    
    else:
        ## Single level
        xts = np.array(list(range(len(df))))
        xt_group_centers = None
    
    
    ## Plot bars    
    # These are always plotted exactly in the order of `df`
    if df.ndim == 1:
        bars = ax.bar(xts, df.values)
    else:
        bars = ax.bar(xts, df.mean(1).values)
    
    
    ## Add errorbars if provided
    if yerrlo is not None and yerrhi is not None:
        # Above we've asserted that the indices match exactly
        yerr = np.array([
            (df.values - yerrlo.values),
            (yerrhi.values - df.values),
            ])
        bar_container = ax.errorbar(
            xts, df.values, yerr=yerr, ls='none', ecolor='k', lw=1,
            elinewidth=elinewidth)
    
        if unclip_error_bars:
            lc = bar_container.lines[2][0]
            lc.set_clip_on(False)
    else:
        bar_container = None
    
    # Set plot kwargs on each bar
    for bar, (iidx, idx_ser) in zip(bars, df.index.to_frame().iterrows()):
        plot_kwargs = index2plot_kwargs(idx_ser)
        if 'alpha' in plot_kwargs:
            bar.set_alpha(plot_kwargs['alpha'])
        if 'ec' in plot_kwargs:
            bar.set_edgecolor(plot_kwargs['ec'])
        if 'fc' in plot_kwargs:
            bar.set_facecolor(plot_kwargs['fc'])
        if 'lw' in plot_kwargs:
            bar.set_linewidth(plot_kwargs['lw'])
        if 'ls' in plot_kwargs:
            bar.set_linestyle(plot_kwargs['ls'])

    # Plot datapoints
    if df.ndim > 1:
        for n_idx, idx in enumerate(df.index):
            xt = xts[n_idx]
            ax.plot([xt] * df.shape[1], df.loc[idx].values, 
                **datapoint_plot_kwargs)

    
    ## Get labels
    if df.index.nlevels > 1:
        # Drop the top level (group name)
        idx_df = df.index.droplevel().to_frame()
    else:
        idx_df = df.index.to_frame()
    
    # Get label for each xtick
    xtls = []
    for iidx, row in idx_df.iterrows():
        if index2label is None:
            xtl = ''
        else:
            xtl = index2label(row)
        xtls.append(xtl)

    # Set labels
    ax.set_xticks(xts)
    ax.set_xticklabels(xtls, **xtls_kwargs)

    
    ## The group labels
    if df.index.nlevels > 1:
        # Iterate over groups
        # `group_names` was already inferred above
        for group_idx, group_center in zip(group_names, xt_group_centers):
            # Get the name of this group
            if group_index2group_label is None:
                group_name = group_idx
            else:
                group_name = group_index2group_label(group_idx)
    
            # We want to place the text centered on the group name in x,
            # at a certain figure location in y.

            if group_name_fig_ypos is not None:
                # x in data, y in figure
                blended_transform = matplotlib.transforms.blended_transform_factory(
                    ax.transData, ax.figure.transFigure)
            
                # text
                ax.text(group_center, group_name_fig_ypos, group_name, 
                    ha='center', va='center', transform=blended_transform,
                    **group_name_kwargs)
            
            elif group_name_y_offset is not None:
                # deprecated
                # Get ypos in axis coordinates
                text_ypos = ax.get_ylim()[0] - group_name_y_offset * (
                    ax.get_ylim()[1] - ax.get_ylim()[0])
                
                # text in axis coordinates
                ax.text(group_center, text_ypos, group_name, 
                    ha='center', va='center',
                    **group_name_kwargs)
    
    return ax, bar_container