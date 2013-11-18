"""Wrapper functions with boilerplate code for making plots the way I like them
"""

import matplotlib
import numpy as np, warnings
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats
import misc

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

def publication_defaults():
    """Set my defaults for font sizes, possibly more later"""
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    #matplotlib.rcParams['figure.facecolor'] = ''


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
    ny = int(np.ceil(n / float(nx)))
    
    if return_fig:
        return plt.subplots(nx, ny, squeeze=squeeze, **kwargs)
    else:
        return nx, ny

def imshow(C, x=None, y=None, ax=None, 
    extent=None, xd_range=None, yd_range=None,
    cmap=plt.cm.RdBu_r, origin='upper', interpolation='nearest', aspect='auto', 
    axis_call='tight', clim=None, center_clim=False):
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
            xwidth = (xd_range[1] - xd_range[0]) / (C.shape[1] - 1)
        except ZeroDivisionError:
            xwidth = 1.
        try:
            ywidth = (yd_range[1] - yd_range[0]) / (C.shape[0] - 1)
        except ZeroDivisionError:
            ywidth=1.
        extent = (
            xd_range[0] - xwidth/2., xd_range[1] + xwidth/2.,
            yd_range[0] - ywidth/2., yd_range[1] + ywidth/2.)

        # Optionally invert the yd_range
        # Because we specify the `extent` manually, we also need to correct
        # it for origin == 'upper'
        if origin == 'upper':
            extent = extent[0], extent[1], extent[3], extent[2]
    
    # Actual call to imshow
    im = ax.imshow(C, interpolation='nearest', origin=origin,
        extent=extent, aspect=aspect, cmap=cmap)
    
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
        new_clim = list(mlab.prctile(data_a, (100.*(1-trim), 100.*trim)))
    
    # Take into account specified clim
    try:
        if clim[0] is not None:
            new_clim[0] = clim[0]
        if clim[1] is not None:
            new_clim[1] = clim[1]
    except IndexError:
        print "warning: problem with provided clim"
    
    # Optionally center
    if center_clim:
        new_clim = np.max(np.abs(new_clim)) * np.array([-1, 1])
    
    # Set to new value
    for ax in axa.flatten():
        for im in ax.get_images():
            im.set_clim(new_clim)
    
    return new_clim

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
            x = range(len(data))
    else:
        single_trace = False        
        if x is None:
            x = range(len(np.mean(data, axis=axis)))
    
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