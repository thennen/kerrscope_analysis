# Experiment has in plane field with same pulse in both directions
import numpy as np
import os
from matplotlib import pyplot as plt

path = r'\\132.239.170.55\SharableDMIsamples\TaCoFeB(V)MgO'
data = track_all(path, level=None, sigma=2.5, skipfiles=1, repeat_ROI=True)

folders = np.array(data.keys())

# Get average field for analyzed folders
meanfields = []
for d in folders:
    logfile = os.path.join(path, d, 'log.txt')
    if os.path.isfile(logfile):
        fields = np.loadtxt(logfile, usecols=[2], delimiter='\t')
        meanfield = np.mean(fields)
        # find index of this dir in data.keys()
        meanfields.append(meanfield)
meanfields = np.array(meanfields)

# Compute left and right slopes, assuming all extracted x and y values are valid
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

# Extract field direction and field value from Rob's folder names
direction = []
folderfields = []
for f in folders:
    f = os.path.split(f)[-1]
    parts = f.split('_')
    if len(parts) > 1:
        ffield = float(parts[0].strip('mT').replace('neg', '-'))
        direc = parts[1]
    else:
        ffield = np.nan
        direc = np.nan
    folderfields.append(ffield)
    direction.append(direc)

direction = np.array(direction)
folderfields = np.array(folderfield)

# Need to reorder data points so that x is increasing for plots
def orderplot(x, y, *args, **kwargs):
    # rearrange x and y so that x is increasing order
    x = np.array(x)
    y = np.array(y)
    ind = np.argsort(x)
    sx = x[ind]
    sy = y[ind]
    plt.plot(sx, sy, *args, **kwargs)

# Find parent folders that contain kerr data folders
parents = np.vectorize(lambda p: os.path.split(p)[0])
parentnames = parents(folders)

# Plot slopes vs IP field vs pulse direction for each parent dir
for p in np.unique(parentnames):
    pmask = parentnames == p
    Bmask = direction == 'B'
    Amask = direction == 'A'
    pbmask = pmask & Bmask
    pamask = pmask & Amask
    plt.figure(figsize=(12,8))
    #plt.scatter(meanfields, leftslopes, label='leftslope')
    #plt.scatter(meanfields, rightslopes, label='rightslope')
    orderplot(meanfields[pbmask], rightslopes[pbmask], '.-', label='rightslope B')
    orderplot(meanfields[pamask], rightslopes[pamask], '.-', label='rightslope A')
    orderplot(meanfields[pbmask], leftslopes[pbmask], '.-', label='leftslope B')
    orderplot(meanfields[pamask], leftslopes[pamask], '.-', label='leftslope A')

    plt.legend(loc=0)
    plt.ylabel('Slope (Pixels/pulse)')
    plt.xlabel('IP Field (mT)')
    plt.title(p)

plt.show()
