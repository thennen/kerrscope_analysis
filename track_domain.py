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
from os.path import join as pjoin
from os.path import isdir
from shutil import copyfile


def track_domain(imdir, repeat_ROI=False):
    # Make new directory to store result of analysis
    contour_dir = pjoin(imdir, 'contours')
    if not isdir(contour_dir):
        os.mkdir(contour_dir)

    # Find files to analyze
    imfns = filter(os.listdir(imdir), '*[0-9][0-9][0-9].png')
    # imfns= [fn for fn in os.listdir(imdir) if fn.endswith('.png')]
    impaths = [pjoin(imdir, fn) for fn in imfns]
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
    fig, ax = make_fig(np.shape(ims[0]))
    step = max(1, len(ims) / 10)
    for im in ims[::step]:
        ax.imshow(im, alpha=.1, cmap='gray')
    ax.imshow(ims[-1], alpha=.1, cmap='gray')

    if repeat_ROI:
        roi = pjoin(contour_dir, 'ROI.txt')
        if os.path.isfile(roi):
            with open(roi, 'r') as f:
                roi_str = f.readline()
                [(i0, i1), (j0, j1)] = [s.split(':') for s in roi_str.split(',')]
                i0, i1, j0, j1 = int(i0), int(i1), int(j0), int(j1)
        else:
            print('ROI.txt not found')
            repeat_ROI = False

    if not repeat_ROI:
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

    fig.savefig(pjoin(contour_dir, 'overlap.png'), pad_inches='tight')

    x1 = []
    x2 = []
    y1 = []
    y2 = []

    plt.ioff()
    bubbles = []
    for subim, fn in zip(subims, imfns):
        fig, ax = make_fig((di, dj))
        ax.imshow(subim, cmap='gray', interpolation='none')
        filt_subim = gaussian_filter(subim, (1, 1))
        level = (np.max(filt_subim) + np.min(filt_subim)) / 2
        contours = find_contours(filt_subim, level)

        # pick the longest contour.  not always right.
        bubble = max(contours, key=len)
        bubbles.append(bubble)
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
        fig.savefig(pjoin(contour_dir, fn), pad_inches='tight')
        plt.close(fig)

    with open(pjoin(contour_dir, 'ROI.txt'), 'w') as f:
        f.write('{}:{}, {}:{}'.format(i0, i1, j0, j1))

    # Plot all the contours
    fig, ax = make_fig((di, dj))
    ax.invert_yaxis()
    cmap = plt.cm.hsv
    colors = [cmap(q) for q in np.linspace(0, 1, len(bubbles))]
    for bub, c in zip(bubbles, colors):
        ax.plot(bub[:, 1], bub[:, 0], linewidth=1.5, c=c)
    fig.savefig(pjoin(contour_dir, 'all_contours.png'), pad_inches='tight')

    # Add last image
    ax.imshow(subims[-1], cmap='gray')
    fontdict = {'size':8}
    mpvalue = np.mean(subims[-1][-15:, -50:])
    if mpvalue > .5:
        fontdict['color'] = 'black'
    else:
        fontdict['color'] = 'white'
    ax.text(dj - 55, di - 3, 'T Hennen', fontdict=fontdict)
    fig.savefig(pjoin(contour_dir, 'all_contours2.png'), pad_inches='tight')

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
    fig1.savefig(pjoin(plotdir, 'left_right.png'), pad_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(imnums, top, '.-', label='Top', c='ForestGreen', linewidth=2)
    ax2.plot(imnums, bottom, '.-', label='Bottom', c='Goldenrod', linewidth=2)
    ax2.set_xlabel('File Number')
    ax2.set_ylabel('Location (pixels)')
    plt.legend(loc=0)
    fig2.savefig(pjoin(plotdir, 'top_bottom.png'), pad_inches='tight')
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(imnums[1:], difftop, '.-', label='Top', c='ForestGreen', linewidth=2)
    ax3.plot(imnums[1:], diffbottom, '.-', label='Bottom', c='Goldenrod', linewidth=2)
    ax3.set_xlabel('File Number')
    ax3.set_ylabel('$\\Delta$ Location (pixels)')
    plt.legend(loc=0)
    fig3.savefig(pjoin(plotdir, 'diff_top_bottom.png'), pad_inches='tight')
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    ax4.plot(imnums[1:], diffleft, '.-', label='Left', c='SlateBlue', linewidth=2)
    ax4.plot(imnums[1:], diffright, '.-', label='Right', c='Crimson', linewidth=2)
    ax4.set_xlabel('File Number')
    ax4.set_ylabel('$\\Delta$ Location (pixels)')
    plt.legend(loc=0)
    fig4.savefig(pjoin(plotdir, 'diff_left_right.png'), pad_inches='tight')
    plt.close(fig4)

    plt.ion()


def write_data((imnums, x1, x2, y1, y2), datadir):
    ''' Write txt files '''
    pdir = os.path.split(datadir)[-1]
    with open(pjoin(datadir, pdir + '_contour_edges.txt'), 'w') as f:
        f.write('image_num\tleft\tright\ttop\tbottom\n')
        fmt = ['%d', '%.2f', '%.2f', '%.2f', '%.2f']
        np.savetxt(f, zip(imnums, x1, x2, y1, y2), delimiter='\t', fmt=fmt)


def analyze_all(dir=r'\\132.239.170.55\SharableDMIsamples\H31', skip=0, **kwargs):
    '''
    Analyze all subdirs, writing plots and extracted data to that subdir.
    Also write summary to parent dir
    '''
    analysis_dir = pjoin(dir, 'Analysis')
    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)
    with open(pjoin(analysis_dir, 'contour_edges.txt'), 'w') as summary_file:
        summary_file.write('image_num\tleft\tright\ttop\tbottom\n')
        fmt = ['%d', '%.2f', '%.2f', '%.2f', '%.2f']
        folders = [f for f in os.listdir(dir)[skip:] if isdir(pjoin(dir, f))]
        for folder in folders:
            if folder != 'Analysis':
                path = pjoin(dir, folder)
                dout = track_domain(path, **kwargs)
                write_plots(dout, path)
                write_data(dout, path)
                # When done, collect some images and data into analysis directory
                summary_file.write('#{}\n'.format(folder))
                np.savetxt(summary_file, zip(*dout), delimiter='\t', fmt=fmt)
                allcontours = pjoin(dir, folder, 'contours', 'all_contours2.png')
                copyfile(allcontours, pjoin(analysis_dir, folder+'_contours.png'))
                plt.close()


def make_fig(shape, dpi=96.):
    ''' return (fig, ax), without axes or white space (maybe)'''
    h, w = shape
    dpi = float(dpi)
    fig = plt.figure()
    fig.set_size_inches(w/dpi, h/dpi, forward=True)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

if __name__ == '__main__':
    pass
    #imdir = r"Tolley DMI PtCoIrPt\Ir 2A\1"
    #imdir= r'C:\Users\thenn\Desktop\bubbles\Tolley DMI PtCoIrPt\Ir 4A\15'
    #track_domain(imdir)
