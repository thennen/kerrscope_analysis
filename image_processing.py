''' Some data analysis functions for kerr microscope images
- highlight differences between two images
- rescale images for contrast
'''

from matplotlib.image import imread
from matplotlib.image import imsave
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
import fnmatch
import heapq
from os.path import join as pjoin
from matplotlib.colors import ListedColormap

# datadir = r'\\132.239.170.55\Tyler'
datadir = r'C:\Users\thenn\Documents\kerrscope_data'
robdir = r'\\132.239.170.55\Tolley InPlane PtCo'

###### Data analysis ######

def diff(im1, im2, bg=None, sigmas=4, thresh=None):
    ''' Find difference of images above the noise. '''

    #im1full = im1
    #im2full = im2

    im1 = im1[:512]
    im2 = im2[:512]

    if bg is not None:
        im1 = im1 - bg[:512]
        im2 = im2 - bg[:512]

    delta = im2 - im1

    if thresh is None:
        def gaus(x, a, sigma, x0):
            return a*np.exp(-(x - x0)**2 / (2 * sigma**2))

        # Do histogram of 1-99 percentile
        # binwidth = 1/512.
        maxpixel = np.max(im2)
        binwidth = maxpixel/2**find_bitdepth(im2)
        hrange = (np.percentile(delta, 1), np.percentile(delta, 99))
        hbinstart = int(hrange[0] / binwidth) * binwidth - binwidth/2
        hbins = np.arange(hbinstart, hrange[1], step=binwidth)
        hval, hbin = np.histogram(delta.flatten(), bins=hbins, range=hrange)
        # Convert to mean value of bin
        hbin = hbin[:-1]/2 + hbin[1:]/2
        popt, pcov = curve_fit(gaus, hbin, hval, p0=[np.max(hval), np.std(delta), 0])

        mean = popt[2]
        sigma = popt[1]
        thresh = (mean - sigmas*sigma, mean + sigmas*sigma)

        # Do a plot to show fit result
        # plt.figure()
        # plt.plot(hbin, hval, '.')
        # hbinsfull = np.arange(-maxpixel - binwidth/2, maxpixel + binwidth/2, step=binwidth)
        # hbinsavg = hbinsfull[:-1]/2 + hbinsfull[1:]/2
        # plt.hist(delta.flatten(), bins=hbinsfull)
        # plt.plot(hbinsavg, gaus(hbinsavg, *popt))
        # plt.vlines((thresh[0], thresh[1]), 0, popt[0], colors='red')
        # plt.xlim((-1, 1))


    delta[(delta > thresh[0]) & (delta < thresh[1])] = np.nan

    return delta


def kerr_hist(fp, bitdepth=9):
    ''' Do histogram for image with certain bit depth.  1 bin per possible pixel
    value '''

    imfull = imread(fp)[:512]
    im = imfull[:512]

    # Don't know why maximum value is 0.5
    binwidth = 0.5/(2**bitdepth)
    hbins = np.arange(-binwidth/2, 0.5 + binwidth/2, step=binwidth)
    plt.figure()
    plt.hist(im.flatten(), bins=hbins)


def find_bitdepth(im):
    maxv = np.max(im)
    #print('maximum pixel value: {}'.format(maxv))
    bitdiffs = np.diff((np.unique(im)))
    bitdiffs = bitdiffs[bitdiffs > 2**-12]
    minbd = min(bitdiffs)
    # Average all differences near the lowest value
    bitdepth = np.log(maxv / np.mean(bitdiffs[bitdiffs - minbd <= 2**-12]))/np.log(2)
    # print('max pixel value: {}, bit depth {}'.format(np.max(im), bitdepth))
    return bitdepth


def diff_folder(folder, **kwargs):
    ''' highlight differences between all the images in a folder '''
    plt.ioff()
    diff_dir = pjoin(folder, 'hl_diffs')
    if not os.path.isdir(diff_dir):
        os.makedirs(diff_dir)
    ims, fps = load_dir(folder)
    fns = [os.path.split(fp)[-1] for fp in fps]
    # contrast stretch everything
    ims = [scale_image(im) for im in ims]
    deltas = []
    for im1, im2, fn in zip(ims[:-1], ims[1:], fns[1:]):
        deltas.append(diff(im1, im2, **kwargs))
        fig, ax = imshow_diff(im1, im2, **kwargs)
        fp = pjoin(diff_dir, fn)
        fig.savefig(fp)
        print('wrote {}'.format(fp))
        plt.close(fig)

    # Do rainbow thing with all the diffs
    cmap = plt.cm.hsv
    colors = [cmap(q) for q in np.linspace(0, 1, len(deltas))]
    fig, ax = make_fig(np.shape(ims[0]))
    for delta, c in zip(deltas, colors):
        cm = ListedColormap(c)
        ax.imshow(delta, cmap=cm)

    fig.savefig(pjoin(diff_dir, 'composite.png'))

    plt.ion()



def diff_everything(folder, **kwargs):
    dirs = [pjoin(folder, d) for d in os.listdir(folder) if d != 'Analysis']
    dirs = [d for d in dirs if os.path.isdir(d)]
    for d in dirs:
        diff_folder(d, **kwargs)

    '''
    #OTHER FUNCTION
    dirs = [dir for dir in os.listdir(parentpath) if os.path.isdir(pjoin(parentpath, dir))]
    for dir in dirs:
        im_dir = pjoin(parentpath, dir)
        rescaled_dir = pjoin(im_dir, 'Rescaled')
        if os.path.isdir(rescaled_dir):
            already_done = os.listdir(rescaled_dir)
        else:
            already_done = []
        fns = fnmatch.filter(os.listdir(im_dir), '*[0-9].png')
        pngs = [fn for fn in fns if fn.endswith('.png')]
        rescale_fns = [fn for fn in pngs if fn not in already_done]
        rescale_fps = [pjoin(im_dir, fn) for fn in rescale_fns]
        ims = [imread(pjoin(im_dir, f)) for f in rescale_fps]
        print('Writing {} new files to {}'.format(len(ims), rescaled_dir))
        write_ims_scaled(ims, rescaled_dir, rescale_fns)
        '''



def scale_image(im, lp=1, hp=99):
    vmin = np.percentile(im[:512], 1)
    vmax = np.percentile(im[:512], 99)
    return (im.clip(vmin, vmax) - vmin) * 0.5 / (vmax - vmin)


###### Make plots ######

def imshow_scaled(im, lp=1, hp=99, cmap='gray', **kwargs):
    vmin = np.percentile(im[:512], lp)
    vmax = np.percentile(im[:512], hp)
    plt.imshow(im, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax, **kwargs)


def make_fig(shape, dpi=96.):
    ''' return (fig, ax), without axes or white space '''
    h, w = shape
    dpi = float(dpi)
    fig = plt.figure()
    fig.set_size_inches(w/dpi, h/dpi, forward=True)
    ax = plt.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)

    return fig, ax


def imshow_diff(im1, im2, bg=None, sigmas=4, thresh=None, cmap='seismic_r', alpha=.4, **kwargs):
    ''' Show image with differences above the noise highlighted '''

    fig, ax = make_fig(np.shape(im2))
    imshow_scaled(im2, lp=1, hp=99)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    delta = diff(im1, im2, bg, sigmas, thresh)

    disnan = np.isnan(delta)
    if np.any(~disnan):
        vmin = np.percentile(delta[~disnan], 1)
        vmax = np.percentile(delta[~disnan], 99)
        # Use range symmetric around zero
        vmax = max((abs(vmin), abs(vmax)))
        vmin = -vmax
    else:
        vmin = vmax = None

    plt.imshow(delta, cmap=cmap, interpolation='none', alpha=alpha, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return fig, ax


###### Import data #####

def load_files(filter):
    ''' load all files matching filter. Searches recursively in datadir '''
    matches = findfile(filter, n=100, returnall=True)
    matches = np.sort(matches)
    print('loading files:')
    for f in matches:
        print(f)
    ims = [imread(f) for f in matches]
    return ims, matches


def load_dir(dir, filter='[0-9][0-9][0-9].png'):
    filter = '*' + filter + '*'
    files = [f for f in os.listdir(dir) if f.endswith('.png')]
    matches = fnmatch.filter(files, filter)
    matches = [pjoin(dir, fn) for fn in np.sort(matches)]
    print('loading files:')
    for f in matches:
        print(f)
    ims = [imread(pjoin(dir, f)) for f in matches]
    return ims, matches


def ipath(path, filter=''):
    ''' Iterate recursively through files in path with specified extension.
    Yield the entire path.
    '''
    if not os.path.isdir(path):
        raise Exception('{} is not a valid directory'.format(path))

    filter = '*' + filter + '*'

    for path, dir, files in os.walk(path):
        # get the whole path, so subdirectory may be specified in the filter
        fpath = [pjoin(path, fname) for fname in files]
        for f in fnmatch.filter(fpath, filter):
            yield pjoin(path, f)


def findfile(filter='', n=1, returnall=False, ext='png'):
    '''
    Return the path of the latest file modified in datadir, which contains
    filter. Wildcards allowed. Case insensitive.
    n may be increased specify earlier files.
    '''
    n = int(n)
    if not os.path.isdir(datadir):
        raise Exception('datadir is not a valid directory')
    filter = '*{}*{}'.format(filter, ext)
    # Look for most recent data file(s)
    try:
        # This goes through all the subdirectories, takes a long time for
        # large directory trees.
        #
        print datadir
        nlargest = heapq.nlargest(n, ipath(datadir, filter), key=os.path.getmtime)

        # This just looks in datadir, not subdirectories
        #
        #fnames = fnmatch.filter(os.listdir(datadir), filter)
        #fpaths = [pjoin(datadir, fn) for fn in fnames]
        #nlargest = heapq.nlargest(n, fpaths, key=os.path.getmtime)
        if n == 1 or not returnall:
            return nlargest[-1]
        return nlargest
    except:
        raise Exception('Fewer than {} filenames match the specified filter'.format(n))


###### Write data #####

def write_ims(ims, dir, fnames, cmap='gray', **kwargs):
    ''' Write a list of ims to disk.  kwargs pass through to mpl.image.imsave'''
    mkdir(dir)

    for im, fn in zip(ims, fnames):
        imsave(pjoin(dir, fn), im, cmap=cmap, **kwargs)


def write_ims_scaled(ims, dir, fnames, lp=1, hp=99, samerange=False, cmap='gray', **kwargs):
    ''' Write a list of ims to disk, scaling to percentile limits '''
    mkdir(dir)

    if samerange:
        alldata = np.concatenate([im[:512].flatten() for im in ims])
        vmin = np.percentile(alldata, lp)
        vmax = np.percentile(alldata, hp)

    for im, fn in zip(ims, fnames):
        if not samerange:
            vmin = np.percentile(im[:512], lp)
            vmax = np.percentile(im[:512], hp)
        imsave(pjoin(dir, fn), im, vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

###### Read and write ######


def rescale_ims(dir, filter='', subdir='Rescaled', lp=1, hp=99, **kwargs):
    '''
    Read .png files from disk, rescale them to percentile limits, and write
    to subdirectory
    '''
    ims, paths = load_dir(dir, filter)

    subdirpath = pjoin(dir, subdir)
    fnames = [os.path.split(fp)[1] for fp in paths]
    write_ims_scaled(ims, subdirpath, fnames, lp=lp, hp=hp, **kwargs)
    print('Wrote scaled images to {}'.format(subdirpath))


def rescale_everything(parentpath=robdir, filter='*[0-9].png'):
    '''
    Rescale all .png in 1st level subdirectories if rescaled file doesn't already exist
    '''
    dirs = [dir for dir in os.listdir(parentpath) if os.path.isdir(pjoin(parentpath, dir))]
    for dir in dirs:
        im_dir = pjoin(parentpath, dir)
        rescaled_dir = pjoin(im_dir, 'Rescaled')
        if os.path.isdir(rescaled_dir):
            already_done = os.listdir(rescaled_dir)
        else:
            already_done = []
        fns = fnmatch.filter(os.listdir(im_dir), '*[0-9].png')
        pngs = [fn for fn in fns if fn.endswith('.png')]
        rescale_fns = [fn for fn in pngs if fn not in already_done]
        rescale_fps = [pjoin(im_dir, fn) for fn in rescale_fns]
        ims = [imread(pjoin(im_dir, f)) for f in rescale_fps]
        print('Writing {} new files to {}'.format(len(ims), rescaled_dir))
        write_ims_scaled(ims, rescaled_dir, rescale_fns)


def rotate_ims(root, dirs, angles):
    ''' Rotate all the images in ~/Rescaled and save to ~/Rotated '''
    for d, a in zip(dirs, angles):
        path = pjoin(root, d)
        rotated_dir = pjoin(path, 'Rotated')
        mkdir(rotated_dir)
        rescale_dir = pjoin(path, 'Rescaled')
        files = [f for f in os.listdir(rescale_dir) if f.endswith('png')]
        for f in files:
            if not os.path.isfile(pjoin(rotated_dir, f)):
                Image.open(pjoin(rescale_dir, f)).rotate(a).save(pjoin(rotated_dir, f))

