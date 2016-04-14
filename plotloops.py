import numpy as np
from matplotlib import pyplot as plt
import os

datadir = r'\\132.239.170.55\Users\User\Desktop\Tolley Jimmy\Gradient Growth Samples\B'
loop_fn = 'OOP analysis.txt'

#sns = os.listdir(datadir)
#for sn in sns:
    # people don't know how to use standard naming conventions so this doesn't
    # work
    #oop_fp = os.path.join(datadir, sn, 'OOP Loop', 'OOP analysis.txt')
paths = []
# Find any file inside each directory with name 'OOP analysis.txt'
for root, dirs, files in os.walk(os.path.join(datadir)):
    if loop_fn in files:
        paths.append(root)

loops = []
for p in paths:
    loop_path = os.path.join(p, loop_fn)
    l = np.loadtxt(loop_path, delimiter='\t', usecols=(0,1), skiprows=1)
    loops.append(l)

fig, ax = plt.subplots()
for path, loop in zip(paths, loops):
    label = '{}\\{}'.format(*path.split('\\')[-3:-1])
    ax.plot(loop[0], loop[1], label=label)

plt.show()
