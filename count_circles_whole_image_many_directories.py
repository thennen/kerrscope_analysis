# run track_domain -i first
from fnmatch import filter
import os

#imdir = r"\\132.239.170.55\SharableDMIsamples\circle_counting\dots_pinch_off"
#imdir = r"\\132.239.170.55\SharableDMIsamples\circle_counting\dots_appear"
source_dir = r'\\132.239.170.55\SharableDMIsamples\circle_counting'
os.listdir(source_dir)
imdirs = filter(os.listdir(source_dir), '*[0-9][0-9]')
imdirs = [pjoin(source_dir, imdir) for imdir in imdirs]

isdir = os.path.isdir
skipfiles = 0
sigma = 1.5

# Keep track of these things
NCIRCLES = []
NNOTCIRCLES = []


for imdir in imdirs:
    ''' Try to track multiple domains'''

    print('Analyzing ' + imdir)
    assert isdir(imdir)
    # Make new directory to store result of analysis
    contour_dir = pjoin(imdir, 'Analysis')
    if not isdir(contour_dir):
        os.mkdir(contour_dir)

    # Find files to analyze
    imfns = kerrims(imdir)
    if len(imfns) <= skipfiles:
        # found nothing in directory
        pass
    # imfns= [fn for fn in os.listdir(imdir) if fn.endswith('.png')]
    impaths = [pjoin(imdir, fn) for fn in imfns]
    ims = [imread(fp) for fp in impaths]
    imnums = [p.split('_')[-1][:-4] for p in impaths]

    # Do contrast stretching for all ims
    ims = [stretch(im[:512], 10, 90) for im in ims]
    # get rid of 3rd channel or whatever (if there is one)
    ims = [im[:,:,0] for im in ims]

    # Whoever wrote kerr program is a goddamn idiot
    # Actually he's a pretty alright guy.
    def fix_shit(astring):
        try:
            return int(astring)
        except:
            return 0
    imnums = map(fix_shit, imnums)

    subims = [im[:512] for im in ims]

    subims = subims[skipfiles:]
    imfns = imfns[skipfiles:]
    imnums = imnums[skipfiles:]
    numcircles = []
    numnotcircles = []
    for subim, fn in zip(subims, imfns):
        # For some reason you must plot contours before imshow for correct png
        # output??
        fig, ax = make_fig((di, dj))
        ax.invert_yaxis()
        filt_subim = gaussian(subim, (2, 2)) - gaussian(subim, (30, 30))
        # Use the middle of the scaled pixel range for threshold level
        pmin = np.min(filt_subim)
        pmax = np.max(filt_subim)
        #level = (np.max(filt_subim) + np.min(filt_subim)) / 2
        #level = np.median(filt_subim)
        #level = np.mean(filt_subim)
        #level = np.percentile(filt_subim, 1)
        level = pmin + 0.3 * (pmax - pmin)
        # Find all the contours
        contours = find_contours(filt_subim, level)
        circles, notcircles = find_circles(contours, sigma=3, smallest_r=3, largest_r=15)
        numcircles.append(len(circles))
        numnotcircles.append(len(notcircles))

        # Write plot of contours
        #ccycle = np.array(['red', 'lime', 'blue', 'yellow', 'orange', 'black'])
        for c in circles:
            ax.plot(c[:, 1], c[:, 0], linewidth=1.5, c='lime')
        for nc in notcircles:
            # Don't even plot small shit
            if (np.max(nc[:,1]) - np.min(nc[:,1]) > 5) and (np.max(nc[:,0]) - np.min(nc[:,0]) > 5):
                ax.plot(nc[:, 1], nc[:, 0], linewidth=1.5, c='red')

        ax.imshow(filt_subim, cmap='gray', interpolation='none')
        fig.savefig(pjoin(contour_dir, fn), pad_inches='tight')
        print('Wrote ' + pjoin(contour_dir, fn))
        plt.close(fig)

    # Make plot of numcircles
    fig, ax = plt.subplots()
    ax.plot(numcircles)
    ax.set_xlabel('Frame #')
    ax.set_ylabel('Number of circles')
    fig.savefig(pjoin(contour_dir, 'numcircles.png'), pad_inches='tight')
    plt.close(fig)

    NCIRCLES.append(numcircles)
    NNOTCIRCLES.append(numnotcircles)

    # regret not saving all the contours

    endpts = [[np.nan , np.nan], [-50, -15], [-50, -14], [-50, -13], [-50, -12], [-50, -11],[-100, -15], [-100, -14], [-100, -13], [-100, -12], [-100, -11], [-100, -10], [0, -15], [0, -14], [0, -13], [0, -12], [0, -11], [0 , -10]]
    start, end = zip(*endpts)
    start = np.array(start)
    end = np.array(end)
    NCIRCLES = np.array(NCIRCLES)

    # Write file

    # make plots
    startvals = unique(start)
    startvals = startvals[~np.isnan(startvals)]
    for s in startvals:
        plt.figure()
        mask = start == s
        num = sum(mask)
        colors = cmap(linspace(0, 1, num))
        for e, numcircs, c in zip(end[mask], NCIRCLES[mask], colors):
            # Plot the raw extracted number
            plot(numcircs, linewidth=1, alpha=.3, c=c)
            # Plot a smooth version
            wwidth = 8.
            window = np.ones(wwidth)/wwidth
            plot(np.convolve(numcircs, window, mode='full'), label=str(e), linewidth=1.5, alpha=1, c=c)
        legend(title='End Field (Oe)', loc=0)
        title('Start Field {} Oe'.format(s))
        xlabel('Time (s)')
        ylabel('Number of circles')
        plt.savefig(pjoin(source_dir, '{}Oe_start.png'.format(s)), bbox_inches='tight')


