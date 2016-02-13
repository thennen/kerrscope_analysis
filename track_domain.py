'''
Track the location of a domain
'''

from select_rect import SelectRect
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.measure import find_contours
from skimage.filters import gaussian_filter

imdir = r"Tolley DMI PtCoIrPt\Ir 2A\1"
contour_dir = os.path.join(imdir, 'contours')
if not os.path.isdir(contour_dir):
    os.mkdir(contour_dir)
#imdir = r"C:\Users\thenn\Desktop\bubbles\Tolley DMI PtCoIrPt\Ir 4A\13"
imfns= [fn for fn in os.listdir(imdir) if fn.endswith('.png')]
impaths = [os.path.join(imdir, fn) for fn in imfns]
ims = [imread(fp) for fp in impaths]
imnums = [p.split('_')[-1][:-4] for p in impaths]

# Whoever wrote kerr program is a goddamn idiot
def fix_shit(astring):
    try:
        return int(astring)
    except:
        return 0
imnums = map(fix_shit, imnums)

# Plot 10 of the images on top of eachother for area selection
fig, ax = plt.subplots()
step = max(1, len(ims) / 10)
for im in ims[::step]:
    ax.imshow(im, alpha=.1, cmap='gray')

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


for subim, fn in zip(subims, imfns):
    plt.ioff()
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
    ax.plot(bubblex, bubbley, linewidth=2, c='Lime')
    ax.hlines((y1[-1], y2[-1]), 0, dj, linestyles='dashed')
    ax.vlines((x1[-1], x2[-1]), 0, di, linestyles='dashed')
    fig.savefig(os.path.join(contour_dir, fn))
    plt.close(fig)
    plt.ion()

fig2, ax2 = plt.subplots()
ax2.plot(imnums, x1, '.-', label='xleft')
ax2.plot(imnums, x2, '.-', label='xright')
ax2.plot(imnums, y1, '.-', label='ytop')
ax2.plot(imnums, y2, '.-', label='ybottom')
ax2.set_xlabel('File Number')
ax2.set_ylabel('Pixel location')
plt.legend(loc=0)

plt.show()
