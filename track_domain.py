'''
Track the location of a domain

TODO:
Trim white space from saved contour plots
Save ROI info so it can be reused
use different ROI shapes
Could add tracking for multiple domains, and average them
'''

from select_rect import SelectRect
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.measure import find_contours
from skimage.filters import gaussian_filter
from skimage import exposure
from fnmatch import filter


def track_domain(imdir):
    # Make new directory to store result of analysis
    contour_dir = os.path.join(imdir, 'contours')
    if not os.path.isdir(contour_dir):
        os.mkdir(contour_dir)

    # Find files to analyze
    imfns = filter(os.listdir(imdir), '*[0-9][0-9][0-9].png')
    #imfns= [fn for fn in os.listdir(imdir) if fn.endswith('.png')]
    impaths = [os.path.join(imdir, fn) for fn in imfns]
    ims = [imread(fp) for fp in impaths]
    imnums = [p.split('_')[-1][:-4] for p in impaths]

    # Do contrast stretching for all ims
    ims = [stretch(im[:512]) for im in ims]

    # Whoever wrote kerr program is a goddamn idiot
    def fix_shit(astring):
        try:
            return int(astring)
        except:
            return 0
    imnums = map(fix_shit, imnums)

    # Plot 10 of the images on top of eachother for area selection
    print('Plotting from {}'.format(imdir))
    fig, ax = plt.subplots()
    step = max(1, len(ims) / 10)
    for im in ims[::step]:
        ax.imshow(im, alpha=.1, cmap='gray')
    ax.imshow(ims[-1], alpha=.1, cmap='gray')

    a = SelectRect(ax)
    plt.show()
    while a.x is None:
        plt.pause(.1)

    # Change to matrix notation
    i0, i1 = a.y[0], a.y[1]
    j0, j1 = a.x[0], a.x[1]
    di = i1 - i0
    dj = j1 - j0
    subims = [im[i0:i1, j0:j1] for im in ims]

    x1 = []
    x2 = []
    y1 = []
    y2 = []

    plt.ioff()
    for subim, fn in zip(subims, imfns):
        fig, ax = make_fig((di, dj))
        ax.imshow(subim, cmap='gray', interpolation='none')
        filt_subim = gaussian_filter(subim, (1,1))
        level = (np.max(filt_subim) + np.min(filt_subim)) / 2
        contours = find_contours(filt_subim, level)

        bubble = max(contours, key=len)
        bubblex = bubble[:, 1]
        bubbley = bubble[:, 0]

        x1.append(min(bubblex))
        x2.append(max(bubblex))
        y1.append(min(bubbley))
        y2.append(max(bubbley))

        # Write plot of contour for reference
        ax.plot(bubblex, bubbley, linewidth=2, c='Lime')
        ax.hlines((y1[-1], y2[-1]), 0, dj, linestyles='dashed')
        ax.vlines((x1[-1], x2[-1]), 0, di, linestyles='dashed')
        fig.savefig(os.path.join(contour_dir, fn))
        plt.close(fig)
    plt.ion()
    return (imnums, x1, x2, y1, y2)

def stretch(image):
    p2, p98 = np.percentile(image, (2, 98))
    return exposure.rescale_intensity(image, in_range=(p2, p98))

def write_plots((imnums, x1, x2, y1, y2), plotdir):
    ''' Write some plots to disk '''
    plt.ioff()

    # Plot data

    left = abs(x1 - x1[0])
    right = abs(x2 - x2[0])
    top = abs(y1 - y1[0])
    bottom = abs(y2 - y2[0])

    diffleft = np.diff(left)
    diffright = np.diff(right)
    difftop = np.diff(top)
    diffbottom = np.diff(bottom)

    fig1, ax1 = plt.subplots()
    ax1.plot(imnums, left, '.-', label='Left', c='SlateBlue', linewidth=2)
    ax1.plot(imnums, right, '.-', label='Right', c='Crimson', linewidth=2)
    ax1.set_xlabel('File Number')
    ax1.set_ylabel('Location (pixels)')
    plt.legend(loc=0)
    fig1.savefig(os.path.join(plotdir, 'left_right.png'))
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(imnums, top, '.-', label='Top', c='ForestGreen', linewidth=2)
    ax2.plot(imnums, bottom, '.-', label='Bottom', c='Goldenrod', linewidth=2)
    ax2.set_xlabel('File Number')
    ax2.set_ylabel('Location (pixels)')
    plt.legend(loc=0)
    fig2.savefig(os.path.join(plotdir, 'top_bottom.png'))
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(imnums[1:], difftop, '.-', label='Top', c='ForestGreen', linewidth=2)
    ax3.plot(imnums[1:], diffbottom, '.-', label='Bottom', c='Goldenrod', linewidth=2)
    ax3.set_xlabel('File Number')
    ax3.set_ylabel('$\\Delta$ Location (pixels)')
    plt.legend(loc=0)
    fig3.savefig(os.path.join(plotdir, 'diff_top_bottom.png'))
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    ax4.plot(imnums[1:], diffleft, '.-', label='Left', c='SlateBlue', linewidth=2)
    ax4.plot(imnums[1:], diffright, '.-', label='Right', c='Crimson', linewidth=2)
    ax4.set_xlabel('File Number')
    ax4.set_ylabel('$\\Delta$ Location (pixels)')
    plt.legend(loc=0)
    fig4.savefig(os.path.join(plotdir, 'diff_left_right.png'))
    plt.close(fig4)

    plt.ion()

def write_data((imnums, x1, x2, y1, y2), datadir):
    ''' Write txt files '''
    with open(os.path.join(datadir, 'contour_edges.txt'), 'w') as f:
        f.write('image_num\tleft\tright\ttop\tbottom\n')
        fmt = ['%d', '%.2f', '%.2f', '%.2f', '%.2f']
        np.savetxt(f, zip(imnums, x1, x2, y1, y2), delimiter='\t', fmt=fmt)

def analyze_all(dir=r'\\132.239.170.55\SharableDMIsamples\H31'):
    ''' Analyze all subdirs, writing plots and extracted data to that subdir '''
    for folder in os.listdir(dir):
        path = os.path.join(dir, folder)
        dout = track_domain(path)
        write_plots(dout, path)
        write_data(dout, path)

def make_fig(shape, dpi=96.):
    ''' return (fig, ax), without axes or white space (maybe)'''
    h, w = shape
    dpi = float(dpi)
    fig = plt.figure()
    fig.set_size_inches(w/dpi, h/dpi, forward=True)
    ax = plt.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

if __name__ == '__main__':
    imdir = r"Tolley DMI PtCoIrPt\Ir 2A\1"
    #imdir= r'C:\Users\thenn\Desktop\bubbles\Tolley DMI PtCoIrPt\Ir 4A\15'
    track_domain(imdir)
