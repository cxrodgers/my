import matplotlib
import numpy as np, warnings
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
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

def vert_bar(bar_lengths, bar_labels=None, bar_positions=None, ax=None,
    bar_errs=None, bar_colors=None, bar_hatches=None, tick_labels_rotation=90,
    plot_bar_ends='ks'):
    """Vertical bar plot"""
    # Defaults
    if bar_positions is None:
        bar_positions = list(range(len(bar_lengths)))
    bar_centers = bar_positions
    if ax is None:
        f, ax = plt.subplots()
    
    # Make the bar plot
    ax.bar(left=bar_centers, bottom=0, width=.8, height=bar_lengths, 
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

def harmonize_clim_in_subplots(fig=None, axa=None, center_clim=False, trim=1):
    """Set clim to be the same in all subplots in figur
    
    f : Figure to grab all axes from, or None
    axa : the list of subplots (if f is None)
    center_clim : if True, the mean of the new clim is always zero
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
        new_clim = (np.min(all_clim_a[:, 0]), np.max(all_clim_a[:, 1]))
    else:
        # Trim to specified prctile of the image data
        data_l = []
        for ax in axa.flatten():
            for im in ax.get_images():
                data_l.append(np.asarray(im.get_array()).flatten())
        data_a = np.concatenate(data_l)
        
        # New clim
        new_clim = mlab.prctile(data_a, (100.*(1-trim), 100.*trim))
    
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