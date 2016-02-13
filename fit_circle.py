'''
Select an area from which to fit a circle using leastsq.  Doesn't work yet.
'''
from select_rect import SelectRect
from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.image import imread

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
subim = im[i0:i1, j0:j1]
ax2.imshow(subim, interpolation='none')
plt.show()

di = i1 - i0
dj = j1 - j0
ii, jj = np.mgrid[:di, :dj]

def circle(centeri, centerj, radius, hlevel=1, llevel=0):
    '''Generate array for a circle '''
    circ = hlevel * np.ones((di, dj))
    circ[(jj - centerj)**2 + (ii - centeri)**2 <= radius**2] = llevel
    return circ

def error((ci, cj, r, h, l)):
    return (subim - circle(ci, cj, r, h, l)).flatten()

#ax2.imshow(circle(di/2, dj/2, dj/4, 1, 0), alpha=.3)
p = leastsq(error, (di/2, dj/2, 35, .25, .1))[0]
print p
cpatch = Circle((p[1], p[0]), radius=p[2], alpha=.3)
ax2.add_patch(cpatch)
