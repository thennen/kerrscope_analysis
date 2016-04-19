'''
Track the location of a domain

TODO:
use different ROI shapes
Could add tracking for multiple domains, and average them
Parse log files for meta data
import data from txt without running analysis again
collect more graphs in Analysis directory
'''

from select_rect import SelectRect
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.measure import find_contours
from skimage.filters import gaussian_filter
from skimage import exposure
import fnmatch
from os.path import join as pjoin
from os.path import split as psplit
from os.path import isdir
from shutil import copyfile
from heapq import nlargest


def kerrims(imdir):
    # Return filenames of kerr images in directory
    assert isdir(imdir)
    return fnmatch.filter(os.listdir(imdir), '*[0-9][0-9][0-9].png')


def kerrfolders(dir, sublevel=None, skipdone=False):
    ''' Return paths of folders containing kerr images. Can specify subdir level
    '''
    assert isdir(dir)

    # This function defines a legit directory
    def kerrfilt(dirpath):
        ignoredirs = ['Analysis']
        if psplit(dirpath)[-1] in ignoredirs:
            return False
        if fnmatch.fnmatch(dirpath, '*\Analysis\*'):
            return False
        if skipdone and os.path.isfile(pjoin(dirpath, 'Analysis', 'ROI.txt')):
            return False
        if len(kerrims(dirpath)) > 0:
            return True

    # Find all image dirs recursively
    folders = []
    dseps = dir.count(os.path.sep)
    for dirpath, dirnames, filenames in os.walk(dir):
        if sublevel is None:
            if kerrfilt(dirpath):
                folders.append(dirpath)
        elif dirpath.count(os.path.sep) - dseps == sublevel:
            if kerrfilt(dirpath):
                folders.append(dirpath)
            # This stops recursion into deeper directories
            del dirnames[:]

    return folders


def set_ROI(dir, skipdone=True):
    # Look for kerr data recursively, select ROI, write it to a file
    folders = kerrfolders(dir, skipdone=skipdone)

    for imdir in folders:
        contour_dir = pjoin(imdir, 'Analysis')
        # Make new directory to store ROI.txt and image
        if not isdir(contour_dir):
            os.mkdir(contour_dir)
        # Find files to analyze
        imfns = kerrims(imdir)
        impaths = [pjoin(imdir, fn) for fn in imfns]
        ims = [imread(fp) for fp in impaths]

        # Do contrast stretching for all ims
        ims = [stretch(im[:512]) for im in ims]

        # Plot <=10 of the images on top of each other for area selection
        print('Select ROI for {}'.format(imdir))
        fig, ax = make_fig(np.shape(ims[0]))
        step = max(1, len(ims) / 10)
        for im in ims[::step]:
            ax.imshow(im, alpha=.1, cmap='gray', interpolation='none')
        ax.imshow(ims[-1], alpha=.1, cmap='gray', interpolation='none')
        a = SelectRect(ax)
        plt.show()
        while a.x is None:
            plt.pause(.1)
        fig.savefig(pjoin(contour_dir, 'overlap.png'), pad_inches='tight')
        plt.close(fig)

        # Change to matrix notation
        i0, i1 = a.y[0], a.y[1]
        j0, j1 = a.x[0], a.x[1]

        with open(pjoin(contour_dir, 'ROI.txt'), 'w') as f:
            f.write('{}:{}, {}:{}'.format(i0, i1, j0, j1))


def track_domain(imdir, repeat_ROI=False, skipfiles=0, sigma=1):
    ''' Try to find contour of domains in imdir.  Write images '''
    assert isdir(imdir)
    # Make new directory to store result of analysis
    contour_dir = pjoin(imdir, 'Analysis')
    if not isdir(contour_dir):
        os.mkdir(contour_dir)

    # Find files to analyze
    imfns = kerrims(imdir)
    if len(imfns) <= skipfiles:
        # found nothing in directory, return something that doesn't break it
        anan = np.array([np.nan])
        return(anan, anan, anan, anan, anan)
    # imfns= [fn for fn in os.listdir(imdir) if fn.endswith('.png')]
    impaths = [pjoin(imdir, fn) for fn in imfns]
    ims = [imread(fp) for fp in impaths]
    imnums = [p.split('_')[-1][:-4] for p in impaths]

    # Do contrast stretching for all ims
    ims = [stretch(im[:512]) for im in ims]

    # Whoever wrote kerr program is a goddamn idiot
    # Actually he's a pretty alright guy.
    def fix_shit(astring):
        try:
            return int(astring)
        except:
            return 0
    imnums = map(fix_shit, imnums)

    if repeat_ROI:
        print('Repeat ROI from {}'.format(imdir))
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
        # Plot 10 of the images on top of each other for area selection
        print('Plotting from {}'.format(imdir))
        fig, ax = make_fig(np.shape(ims[0]))
        step = max(1, len(ims) / 10)
        for im in ims[::step]:
            ax.imshow(im, alpha=.1, cmap='gray', interpolation='none')
        ax.imshow(ims[-1], alpha=.1, cmap='gray', interpolation='none')
        a = SelectRect(ax)
        plt.show()
        while a.x is None:
            plt.pause(.1)
        fig.savefig(pjoin(contour_dir, 'overlap.png'), pad_inches='tight')

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
    subims = subims[skipfiles:]
    imfns = imfns[skipfiles:]
    imnums = imnums[skipfiles:]
    bubbles = []
    for subim, fn in zip(subims, imfns):
        # For some reason you must plot contours before imshow for correct png
        # output??
        fig, ax = make_fig((di, dj))
        ax.invert_yaxis()
        filt_subim = gaussian_filter(subim, (sigma, sigma))
        level = (np.max(filt_subim) + np.min(filt_subim)) / 2
        contours = find_contours(filt_subim, level)

        # pick the longest contour.  not always right.
        largest_contours = nlargest(2, contours, key=len)
        bubble = largest_contours[0]

        # If contour ends on two different edges of picture, also append second
        # longest
        def edge((i, j)):
            # return what edge the point is on
            if i < 2:
                return 1
            elif i > di - 2:
                return 2
            elif j < 2:
                return 3
            elif j > dj - 2:
                return 4
            else:
                return 0
        i_edge = edge(bubble[0])
        f_edge = edge(bubble[-1])
        if (i_edge and f_edge) and (i_edge != f_edge) and len(largest_contours) > 1:
            bubble = np.vstack((bubble, largest_contours[1]))

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

        ax.imshow(subim, cmap='gray', interpolation='none')
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

    # Add last subim
    ax.imshow(subims[-1], cmap='gray', interpolation='none')
    fontdict = {'size':8}
    # Add watermark.  Pick whether it's white or black
    mpvalue = np.mean(subims[-1][-15:, -50:])
    if mpvalue > .5:
        fontdict['color'] = 'black'
    else:
        fontdict['color'] = 'white'
    ax.text(dj - 55, di - 3, 'T Hennen', fontdict=fontdict)
    fig.savefig(pjoin(contour_dir, 'all_contours2.png'), pad_inches='tight')

    #plt.close(fig)

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

    leftfit = np.polyfit(imnums, left, 1)
    rightfit = np.polyfit(imnums, right, 1)
    #topfit = np.polyfit(imnums, top, 1)
    #bottomfit = np.polyfit(imnums, bottom, 1)

    leftpolyval = np.polyval(leftfit, imnums)
    rightpolyval = np.polyval(rightfit, imnums)
    #toppolyval = np.polyval(topfit, imnums)
    #bottompolyval = np.polyval(bottomfit, imnums)

    leftfitlabel = '{:2f}'.format(leftfit[0])
    rightfitlabel = '{:2f}'.format(rightfit[0])

    diffleft = np.diff(left)
    diffright = np.diff(right)
    difftop = np.diff(top)
    diffbottom = np.diff(bottom)

    fig1, ax1 = plt.subplots()
    ax1.plot(imnums, left, '.-', label='Left', c='SlateBlue', linewidth=2)
    ax1.plot(imnums, leftpolyval, '--', label=leftfitlabel, c='SlateBlue', alpha=.4)
    ax1.plot(imnums, right, '.-', label='Right', c='Crimson', linewidth=2)
    ax1.plot(imnums, rightpolyval, '--', label=rightfitlabel, c='Crimson', alpha=.4)
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


def track_all(dir=r'\\132.239.170.55\SharableDMIsamples\H31', level=None, skip=0, skipdone=False, **kwargs):
    '''
    Analyze all subdirs (subdir level input), writing plots and extracted data
    to that subdir.  Also write summary to parent dir
    '''
    assert isdir(dir)
    data = []
    analysis_dir = pjoin(dir, 'Analysis')
    if not isdir(analysis_dir):
        os.makedirs(analysis_dir)
    with open(pjoin(analysis_dir, 'contour_edges.txt'), 'w') as summary_file:
        summary_file.write('image_num\tleft\tright\ttop\tbottom\n')
        fmt = ['%d', '%.2f', '%.2f', '%.2f', '%.2f']

        folders = kerrfolders(dir, sublevel=level, skipdone=skipdone)

        for folder in folders:
            dout = track_domain(folder, **kwargs)
            data.append(dout)
            write_plots(dout, folder)
            write_data(dout, folder)
            # When done, collect some images and data into analysis directory
            summary_file.write('#{}\n'.format(folder))
            np.savetxt(summary_file, zip(*dout), delimiter='\t', fmt=fmt)
            # Copy contour overlap pngs into Analysis directory
            allcontours = pjoin(folder, 'Analysis', 'all_contours2.png')
            contourdir = pjoin(analysis_dir, 'Contours')
            if not isdir(contourdir):
                os.makedirs(contourdir)
            # Construct filename so that there are no conflicts
            fn_prefix = '_'.join(folder.replace(dir, '').split(os.path.sep)[1:])
            contour_fn = fn_prefix + '_contours.png'
            copyfile(allcontours, pjoin(contourdir, contour_fn))
            # Copy slope graphs
            allcontours = pjoin(folder, 'left_right.png')
            lrslopedir = pjoin(analysis_dir, 'LR_Slopes')
            if not isdir(lrslopedir):
                os.makedirs(lrslopedir)
            slope_fn = fn_prefix + '_LRslope.png'
            copyfile(allcontours, pjoin(lrslopedir, slope_fn))
            plt.close()

    return {f:d for f,d in zip(folders, data)}


def import_data(fp):
    ''' Get results of analysis from file '''
    assert os.path.isfile(fp)


def make_fig(shape):
    ''' return (fig, ax), without axes or white space (maybe)'''
    # For some reason this totally fails if you let the figure be shown before
    # savefig
    h, w = shape
    dpi = 100.
    fig = plt.figure(frameon=False, dpi=dpi)
    fig.set_size_inches(w/dpi, h/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

if __name__ == '__main__':
    pass
    #imdir = r"Tolley DMI PtCoIrPt\Ir 2A\1"
    #imdir= r'C:\Users\thenn\Desktop\bubbles\Tolley DMI PtCoIrPt\Ir 4A\15'
    #track_domain(imdir)
