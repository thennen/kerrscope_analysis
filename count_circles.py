# run track_domain -i first

#imdir = r"\\132.239.170.55\SharableDMIsamples\circle_counting\dots_pinch_off"
#imdir = r"\\132.239.170.55\SharableDMIsamples\circle_counting\dots_appear"
imdir = r'\\132.239.170.55\SharableDMIsamples\circle_counting\CoPtOs2A Skyrmion Formation_68'

isdir = os.path.isdir
repeat_ROI = True
skipfiles = 0
sigma = 2

''' Try to track multiple domains'''
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
ims = [stretch(im[:512]) for im in ims]
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
    filt_subim = gaussian(subim, (sigma, sigma))
    # Use the middle of the scaled pixel range for threshold level
    level = (np.max(filt_subim) + np.min(filt_subim)) / 2
    # Find all the contours
    contours = find_contours(filt_subim, level)
    circles, notcircles = find_circles(contours, sigma=1)
    numcircles.append(len(circles))
    numnotcircles.append(len(notcircles))

    # Write plot of contours
    #ccycle = np.array(['red', 'lime', 'blue', 'yellow', 'orange', 'black'])
    for c in circles:
        ax.plot(c[:, 1], c[:, 0], linewidth=1.5, c='lime')
    for nc in notcircles:
        ax.plot(nc[:, 1], nc[:, 0], linewidth=1.5, c='red')

    ax.imshow(subim, cmap='gray', interpolation='none')
    fig.savefig(pjoin(contour_dir, fn), pad_inches='tight')
    plt.close(fig)

# Write file


