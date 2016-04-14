# Experiment has in plane field with same pulse in both directions
import numpy as np
import os
from matplotlib import pyplot as plt

path = r'\\132.239.170.55\SharableDMIsamples\Expansion CoPtOs1APt'
data = analyze_all(path, level=1, sigma=2.5, skipfiles=1, repeat_ROI=True)

folders = np.array([os.path.split(fp)[-1] for fp in data.keys()])

# Get average field for images
meanfields = np.nan * np.ones(len(data.keys()))
dirs = os.listdir(path)
for d in dirs:
    logfile = os.path.join(path, d, 'log.txt')
    if os.path.isfile(logfile):
        fields = np.loadtxt(logfile, usecols=[2], delimiter='\t')
        meanfield = np.mean(fields)
        # find index of this dir in data.keys()
        meanfields[np.where(folders == d)] = meanfield

leftslopes = []
rightslopes = []
for dtable in data.values():
    imnums = dtable[0]
    x1 = dtable[1]
    x2 = dtable[2]
    left = abs(x1 - x1[0])
    right = abs(x2 - x2[0])
    leftslope, _ = np.polyfit(imnums, left, 1)
    rightslope, _ = np.polyfit(imnums, right, 1)
    leftslopes.append(leftslope)
    rightslopes.append(rightslope)

leftslopes = np.array(leftslopes)
rightslopes = np.array(rightslopes)

folders = np.array([os.path.split(fp)[-1] for fp in data.keys()])

direction = []
folderfield = []
for f in folders:
    parts = f.split(' ')
    if len(parts) == 3:
        ffield = float(parts[0].strip('mT').replace('neg', '-'))
        direc = parts[1]
    else:
        ffield = np.nan
        direc = np.nan
    folderfield.append(ffield)
    direction.append(direc)

direction = np.array(direction)
folderfield = np.array(folderfield)

def orderplot(x, y, *args, **kwargs):
    # rearrange x and y so that x is increasing order
    x = np.array(x)
    y = np.array(y)
    ind = np.argsort(x)
    sx = x[ind]
    sy = y[ind]
    plt.plot(sx, sy, *args, **kwargs)

# Plot slopes vs IP field
Bmask = direction == 'B'
Amask = direction == 'A'
plt.figure()
#plt.scatter(meanfields, leftslopes, label='leftslope')
#plt.scatter(meanfields, rightslopes, label='rightslope')
orderplot(folderfield[Bmask], rightslopes[Bmask], '.-', label='rightslope B')
orderplot(folderfield[Amask], rightslopes[Amask], '.-', label='rightslope A')
orderplot(folderfield[Bmask], leftslopes[Bmask], '.-', label='leftslope B')
orderplot(folderfield[Amask], leftslopes[Amask], '.-', label='leftslope A')

plt.legend(loc=0)
plt.ylabel('Slope (Pixels/pulse)')
plt.xlabel('IP Field (mT)')
