import os
import sys

fileDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.dirname(fileDir)
sys.path.append(rootDir)

import argparse
import tqdm
from pycurve.pycurve import CurveMeasurer
from pycurve.util import videoCapture, getVideoTime

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str,   default=os.path.join(rootDir, 'data/210607_actuator.MOV'), help="video directory")
parser.add_argument("--out", type=str,   default="xx", help="output directory")
args = parser.parse_args()

videoDir = args.video
stemName = ".".join(os.path.basename(videoDir).split('.')[:-1])
outFolderDir = os.path.join(rootDir, 'output/{}/'.format(stemName)) if args.out == "xx" else args.out

setting = {
    'rgbLb': (30, 200, 40),
    'rgbUb': (150, 255, 150),
    'dilateErode': [20],
    'alpha': 2.1,
    'beta': -127,
    'numCurves': 1
}
cm = CurveMeasurer(setting)

textItems = ['No.\ttime\tradius ']

t0 = 2
dt = 10.2
T = getVideoTime(videoDir)

t = t0
for iActuation in tqdm.tqdm(range(int(T // dt + 1))):
    ret = videoCapture(videoDir, t)
    if ret is not False:
        image = ret
        cm.detect(image)
        cm.saveImage(os.path.join(outFolderDir, '{:.2f}.png').format(t))
        textItems.append("{}\t{:.2f}\t{:.2f}".format(iActuation, t, cm.rs[0]))
        # cm.show()
        t += dt
    else:
        break
    
    # break

text = "\n".join(textItems)
with open(os.path.join(outFolderDir, 'data.txt'), 'w') as ofile:
    ofile.write(text)
