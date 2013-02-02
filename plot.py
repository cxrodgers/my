import matplotlib

def font_embed():
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['svg.fonttype'] = 'svgfont'
