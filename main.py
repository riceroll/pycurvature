import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert

inDir = '/Users/Roll/Desktop/scale.png'
nSamples = 8

name = inDir.split('/')[-1].split('.')[0]
suffix = inDir.split('/')[-1].split('.')[-1]
parentDir = "/"+"/".join(inDir.split('/')[:-1])+"/"
outDir = parentDir + name + "_out" + "." + suffix

dl = -5
du = -1

nDilate = 3
nErode = 3
nDilate2 = 0

alpha = 3 # Contrast control (1.0-3.0)

alpha = 2.5 # Contrast control (1.0-3.0)
beta = -127 # Brightness control (0-100)

imgs = []

img = cv2.imread(inDir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img0 = img.copy()
img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

imgs.append(img.copy())
print(img[248,100])
print(img[103,248])
img = cv2.inRange(img, (250 + dl, 150 + dl, 0 + dl), (255-du, 255-du, 95-du))



img = np.array(img, np.uint8)

imgs.append(img.copy())

skel = np.zeros(img.shape, np.uint8)
size = np.size(img)
element = np.ones([3, 3])

for i in range(nDilate):
    img = cv2.dilate(img, element)

for i in range(nErode):
    img = cv2.erode(img, element)


for i in range(nDilate2):
    img = cv2.dilate(img, element)

for i in range(20):
    img = cv2.erode(img, element)
    img = cv2.dilate(img, element)

imgs.append(img.copy())

image = np.array(img, bool)
skeleton = skeletonize(image)
skeleton = np.array(skeleton, np.uint8) * 255
img = skeleton


imgs.append(img.copy())

contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

lengths = np.array([c[:,0,0].max() - c[:,0,0].min() for c in contours])
widths = np.array([c[:,0,1].max() - c[:,0,1].min() for c in contours])
areas = np.array([lengths[i] + widths[i] for i in range(len(lengths))])

args = np.argsort(areas)
args = args[::-1]
args = args[:nSamples]

# for arg in args:
#     c = contours[arg]
#     blank = np.zeros(img.shape, np.uint8)
#     cv2.drawContours(blank, [c], -1, (255,255,255), 1)
#     plt.imshow(blank)
#     plt.show()

from scipy import optimize

xcs = []
ycs = []
rs = []

for arg in args:
    c = contours[arg]
    c = c.squeeze(1)
    
    def distance(input):
        xc, yc, r = input
        ret = (c[:, 0] - xc) ** 2 + (c[:, 1] - yc) ** 2 - r ** 2
        if r < 0:
            ret -= 100000
        return ret
    
    sols = optimize.leastsq(distance, np.array([c.mean(0)[0], c.mean(1)[1], (c.max(0)[0] - c.min(0)[0]) / 2]))
    
    xc, yc, r = sols[0]
    xcs.append(xc)
    ycs.append(yc)
    rs.append(r)
    
    img0 = cv2.circle(img0, [int(xc), int(yc)], int(r), color=(255, 0, 0), thickness=1)
    img0 = cv2.putText(img0, str("{:.2f}".format(r)), (int(xc), int(yc)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                       color=(255, 255, 200), thickness=1, lineType=cv2.LINE_AA)

print(rs)

imgs.append(img0)

height = int(np.ceil(len(imgs) / 3))
width = 3

fig = plt.figure(figsize=(16,9))
for i, img in enumerate(imgs):
    fig.add_subplot(height, width, i+1)
    plt.imshow(img)

plt.show()

plt.imsave(outDir, img0)


