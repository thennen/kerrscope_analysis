''' Some data analysis functions for kerr microscope images '''
from matplotlib.image import imread
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
import fnmatch
import heapq

# datadir = r'\\132.239.170.55\Tyler'
datadir = r'C:\Users\thenn\Documents\kerrscope_data'


def imshow_diff(im1, im2, bg=None, sigmas=4, thresh=None, cmap='seismic_r', alpha=.4, **kwargs):
    ''' Show image with differences above the noise highlighted '''

    delta = diff(im1, im2, bg, sigmas, thresh)

    plt.figure()
    imshow_scaled(im2, lp=1, hp=2)
    vmin = np.percentile(delta[~np.isnan(delta)], 1)
    vmax = np.percentile(delta[~np.isnan(delta)], 99)
    # Use range symmetric around zero
    vmax = max((abs(vmin), abs(vmax)))
    vmin = -vmax

    plt.imshow(delta, cmap=cmap, interpolation='none', alpha=alpha, vmin=vmin, vmax=vmax, **kwargs)

def diff(im1, im2, bg=None, sigmas=4, thresh=None):
    ''' Find difference of images above the noise. '''

    im1full = im1
    im2full = im2

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
        binwidth = 0.5/2**find_bitdepth(im2)
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
        #plt.figure()
        #plt.plot(hbin, hval, '.')
        #hbinsfull = np.arange(-0.5 - binwidth/2, 0.5 + binwidth/2, step=binwidth)
        #hbinsavg = hbinsfull[:-1]/2 + hbinsfull[1:]/2
        #plt.hist(delta.flatten(), bins=hbinsfull)
        #plt.plot(hbinsavg, gaus(hbinsavg, *popt))
        #plt.vlines((thresh[0], thresh[1]), 0, popt[0], colors='red')
        #plt.xlim((-1, 1))


    delta[(delta > thresh[0]) & (delta < thresh[1])] = np.nan

    return delta

def load_files(filter):
    files = findfile(filter, n=100, returnall=True)
    ims = [imread(f) for f in files]
    return sort(ims)

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

def imshow_scaled(im, lp=1, hp=99, cmap='gray', **kwargs):
    vmin = np.percentile(im[:512], 1)
    vmax = np.percentile(im[:512], 99)
    plt.imshow(im, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax, **kwargs)

def find_bitdepth(im):
    bitdiffs = np.diff((np.unique(im)))
    bitdiffs = bitdiffs[bitdiffs > 2**-12]
    minbd = min(bitdiffs)
    # Average all differences near the lowest value
    bitdepth = np.log(0.5 / np.mean(bitdiffs[bitdiffs - minbd <= 2**-12]))/np.log(2)
    # print('max pixel value: {}, bit depth {}'.format(np.max(im), bitdepth))
    return bitdepth

def ipath(path, filter=''):
    ''' Iterate recursively through files in path with specified extension.
    Yield the entire path.
    '''
    if not os.path.isdir(path):
        raise Exception('{} is not a valid directory'.format(path))

    filter = '*' + filter + '*'

    for path, dir, files in os.walk(path):
        # get the whole path, so subdirectory may be specified in the filter
        fpath = [os.path.join(path, fname) for fname in files]
        for f in fnmatch.filter(fpath, filter):
            yield os.path.join(path, f)


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
        nlargest = heapq.nlargest(n, ipath(datadir, filter), key=os.path.getmtime)

        # This just looks in datadir, not subdirectories
        #
        #fnames = fnmatch.filter(os.listdir(datadir), filter)
        #fpaths = [os.path.join(datadir, fn) for fn in fnames]
        #nlargest = heapq.nlargest(n, fpaths, key=os.path.getmtime)
        if n == 1 or not returnall:
            return nlargest[-1]
        return nlargest
    except:
        raise Exception('Fewer than {} filenames match the specified filter'.format(n))


def diff_folder(folder, **kwargs):
    ''' highlight differences between all the images in a folder '''
