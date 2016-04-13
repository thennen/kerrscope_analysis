import numpy as np

path = r'C:\Users\thenn\desktop\H31 Flow Regime Determination'
data = analyze_all(path, level=2, repeat_ROI=True)

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

def parse_idiot(idiot):
    parts = idiot.split('_')
    return float(parts[0].replace('point', '.') + 'E-4')
duration = [parse_idiot(psplit(psplit(p)[0])[1]) for p in data.keys()]
volts = [os.path.split(p)[-1].split('_')[0] for p in data.keys()]

duration = array(duration)
volts = array(volts)
leftslopes = array(leftslopes)
rightslopes = array(rightslopes)

figure()
for dur in unique(duration):
    durmask = duration == dur
    meanslope = (leftslopes[durmask] + rightslopes[durmask]) / 2
    maskvolts = volts[durmask]
    ind = argsort(maskvolts)
    v = maskvolts[ind]
    s = meanslope[ind]
    plot(v, s, '.-', label=str(dur))

legend(title='Pulse Duration (S)')
ylabel('Speed (Pixels / pulse)')
xlabel('Pulse Voltage (V)')

figure()
for vol in unique(volts):
    volmask = volts == vol
    meanslope = (leftslopes[volmask] + rightslopes[volmask]) / 2
    maskdur = duration[volmask]
    ind = argsort(maskdur)
    d = maskdur[ind]
    s = meanslope[ind]
    plot(d, s, '.-', label=str(vol))

legend(title='Pulse Voltage (V)')
ylabel('Speed (Pixels / pulse)')
xlabel('Pulse Duration (S)')
