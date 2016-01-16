''' Some data analysis functions for kerr microscope images '''
from matplotlib.image import imread
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def imshow_diff(fp1, fp2, bgfp=None, sigmas=4, thresh=None):
    ''' Show image with differences above the noise highlighted '''


    im1full = imread(fp1)[:512]
    im1 = im1full[:512]
    im2full = imread(fp2)
    im2 = im2full[:512]

    if bgfp is not None:
        bg = imread(bgfp)[:512]
        im1 = im1 - bg
        im2 = im2 - bg

    diff = im2 - im1

    if thresh is None:
        def gaus(x, a, sigma, x0):
            return a*np.exp(-(x - x0)**2 / (2 * sigma**2))

        # Do histogram of 1-99 percentile
        # binwidth = 1/512.
        binwidth = 0.5/2**find_bitdepth(im2)
        hrange = (np.percentile(diff, 1), np.percentile(diff, 99))
        hbinstart = int(hrange[0] / binwidth) * binwidth - binwidth/2
        hbins = np.arange(hbinstart, hrange[1], step=binwidth)
        hval, hbin = np.histogram(diff.flatten(), bins=hbins, range=hrange)
        # Convert to mean value of bin
        hbin = hbin[:-1]/2 + hbin[1:]/2
        popt, pcov = curve_fit(gaus, hbin, hval, p0=[max(hval), np.std(diff), 0])

        mean = popt[2]
        sigma = popt[1]
        thresh = (mean - sigmas*sigma, mean + sigmas*sigma)

        # Do a plot to show fit result
        plt.figure()
        #plt.plot(hbin, hval, '.')
        hbinsfull = np.arange(-0.5 - binwidth/2, 0.5 + binwidth/2, step=binwidth)
        hbinsavg = hbinsfull[:-1]/2 + hbinsfull[1:]/2
        plt.hist(diff.flatten(), bins=hbinsfull)
        plt.plot(hbinsavg, gaus(hbinsavg, *popt))
        plt.vlines((thresh[0], thresh[1]), 0, popt[0], colors='red')
        plt.xlim((-1, 1))


    diff[(diff > thresh[0]) & (diff < thresh[1])] = np.nan
    plt.figure()
    vmin = np.percentile(im2, 1)
    vmax = np.percentile(im2, 99)
    plt.imshow(im2, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
    vmin2 = np.percentile(im2, 1)
    vmax2 = np.percentile(im2, 99)
    plt.imshow(diff, alpha=.3, cmap='cool', interpolation='none', vmin=vmin2, vmax=vmax2)

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
    bitdiffs = np.diff(np.sort(np.unique(im)))
    bitdiffs = bitdiffs[bitdiffs > 2**-12]
    minbd = min(bitdiffs)
    # Average all differences near the lowest value
    bitdepth = np.log(0.5 / np.mean(bitdiffs[bitdiffs - minbd <= 2**-12]))/np.log(2)
    # print('max pixel value: {}, bit depth {}'.format(np.max(im), bitdepth))
    return bitdepth

def diff_folder(folder, **kwargs):
    ''' highlight differences between all the images in a folder '''
