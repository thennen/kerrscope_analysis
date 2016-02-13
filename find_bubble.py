'''
Select an area in which to detect the edges of one bubble.  Using crappy method
that I made up.

Finds approximate bubble location, but doesn't find size
'''

from select_rect import SelectRect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.image import imread
import scipy.ndimage as ndimage

# Select some pixels from this image
im = imread(r"Tolley DMI PtCoIrPt\Ir 2A\1\20150107PtCoIr(2A)Pt_020.png")

fig, ax = plt.subplots()
ax.imshow(im, cmap='gray', interpolation='none')
a = SelectRect(ax)
plt.show()
while a.x is None:
    plt.pause(.1)
print('x = {}, {}'.format(*a.x))
print('y = {}, {}'.format(*a.y))
fig2, ax2 = plt.subplots()

# Change to matrix notation
i0, i1 = a.y[0], a.y[1]
j0, j1 = a.x[0], a.x[1]

di = i1 - i0
dj = j1 - j0
subim = im[i0:i1, j0:j1]
ax2.imshow(subim, interpolation='none')
plt.show()

# Estimate center of bubble by finding local min/max
# filt_subim = ndimage.gaussian_filter(subim, (2,2))
imean = np.mean(subim, axis=0)
jmean = np.mean(subim, axis=1)
mini, maxi = np.argmin(imean), np.argmax(imean)
minj, maxj = np.argmin(jmean), np.argmax(jmean)

# Just use whichever extreme is closer to the center of the image ??
# maxdist = (maxi - di/2)**2 + (maxj - dj/2)**2
# mindist = (mini - di/2)**2 + (minj - dj/2)**2
#if maxdist > mindist:
#    center = (mini, minj)
#else:
#    center = (maxi, maxj)

# Use whichever extreme is least like the image edges
edgemean = np.mean(np.concatenate((subim[0,:], subim[:,0], subim[-1,:], subim[:,-1])))
icloser = np.abs(imean[mini] - edgemean) < np.abs(imean[maxi] - edgemean)
jcloser = np.abs(jmean[minj] - edgemean) < np.abs(jmean[maxj] - edgemean)
if icloser and jcloser:
    center = (maxi, maxj)
elif not icloser and not jcloser:
    center = (mini, minj)
else:
    print('Shitty algorithm got confused guessing the center of bubble')

cpatch = Circle(center, radius=10, alpha=.3)
ax2.add_patch(cpatch)
