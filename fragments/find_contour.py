'''
Find contour of domain
'''

from select_rect import SelectRect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import scipy.ndimage as ndimage
from skimage.measure import find_contours
from skimage.filters import gaussian_filter

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
ax2.imshow(subim, cmap='gray', interpolation='none')
plt.show()

filt_subim = gaussian_filter(subim, (1,1))
level = (np.max(filt_subim) + np.min(filt_subim)) / 2
contours = find_contours(filt_subim, level)

bubble = max(contours, key=len)
ax2.plot(bubble[:, 1], bubble[:, 0], linewidth=2, c='Lime')
