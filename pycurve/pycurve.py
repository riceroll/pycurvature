import os
import numpy as np
import cv2
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

class CurveMeasurer:
    """
    example usage:
        image = cv2.imread("../data/foo.png")
        cm = CurveMeasurer(image, setting)
        cm.detect()
        cm.show()
        cm.saveImage("../output/foo.png")
    """
    
    def __init__(self, setting):
        setting = self.loadDefaultSetting(setting)
        
        self.rgbLb = setting['rgbLb']
        self.rgbUb = setting['rgbUb']
        self.dilateErode = setting['dilateErode']       # e.g. [3, -3] means dilate 3 times then erode 3 times
        self.alpha = setting['alpha']       # alpha: contrast control (1.0 - 3.0)
        self.beta = setting['beta']          # beta: contrast control (-127 - 127)
        self.numCurves = setting['numCurves']   # numCurve: expected number of curves in the photo
        
        self.imgOut = None  # np.array  output image with labels of curvature
        self.images = []    # [] of np.array to display
        self.rs = []      # [] of curvature radius
    
    @staticmethod
    def loadDefaultSetting(setting):
        defaultSetting = {
            'rgbLb': (250, 150, 0),
            'rgbUb': (255, 255, 95),
            'dilateErode': [2, -2],
            'alpha': 2.5,
            'beta': -127,
            'numCurves': 10
        }
        for key in defaultSetting:
            if key not in setting:
                setting[key] = defaultSetting[key]
        return setting
    
    def detect(self, image):
        """
        detect curvatures, store the imgs and radius, label the curvatures
        
        image: np.array input image by cv2.imread() in BGR color
        :return:
            self.imgs
            self.imgOut
            self.rs
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img0 = img.copy()
        img = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
        self.images.append(img.copy())
        
        img = cv2.inRange(img, self.rgbLb, self.rgbUb)
        img = np.array(img, np.uint8)
        self.images.append(img.copy())

        element = np.ones([3, 3])

        for n in self.dilateErode:
            if n > 0:
                for i in range(n):
                    img = cv2.dilate(img, element)
            else:
                for i in range(-n):
                    img = cv2.erode(img, element)

        for i in range(20):
            img = cv2.erode(img, element)
            img = cv2.dilate(img, element)
        self.images.append(img.copy())

        image = np.array(img, bool)
        skeleton = skeletonize(image)
        skeleton = np.array(skeleton, np.uint8) * 255
        img = skeleton
        self.images.append(img.copy())

        contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lengths = np.array([c[:, 0, 0].max() - c[:, 0, 0].min() for c in contours])
        widths = np.array([c[:, 0, 1].max() - c[:, 0, 1].min() for c in contours])
        areas = np.array([lengths[i] + widths[i] for i in range(len(lengths))])

        args = np.argsort(areas)
        args = args[::-1]
        args = args[:self.numCurves]

        # for arg in args:
        #     c = contours[arg]
        #     blank = np.zeros(img.shape, np.uint8)
        #     cv2.drawContours(blank, [c], -1, (255,255,255), 1)
        #     plt.imshow(blank)
        #     plt.show()

        # optimize the center and radius of curvatures with least square
        from scipy import optimize
        xcs = []
        ycs = []
        rs = []     # radiuses
        for arg in args:
            c = contours[arg]
            center = c.mean(0)
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
    
            fontScale = np.sqrt(img0.shape[0] * img0.shape[1] / (400 * 400)) * 0.5
            thickness = int(np.sqrt(img0.shape[0] * img0.shape[1] / (400 * 400)) * 1)
            
            img0 = cv2.circle(img0, [int(xc), int(yc)], int(r), color=(255, 0, 0), thickness=thickness)
            img0 = cv2.putText(img0, str("{:.2f}".format(r)), (int(center[0,0]), int(center[0,1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=fontScale,
                               color=(255, 50, 200), thickness=thickness, lineType=cv2.LINE_AA)
            
        self.images.append(img0)
        self.imgOut = img0
        self.rs = rs
        
    def show(self):
        height = int(np.ceil(len(self.images) / 3))
        width = 3
    
        fig = plt.figure(figsize=(16, 9))
        for i, img in enumerate(self.images):
            fig.add_subplot(height, width, i + 1)
            plt.imshow(img)
        
        print(self.rs)
        plt.show()
    
    def saveImage(self, dir):
        plt.imsave(dir, self.imgOut)

if __name__ == "__main__":
    fileDir = os.path.dirname(os.path.realpath(__file__))
    rootDir = os.path.dirname(fileDir)
    
    inDir = os.path.join(rootDir, 'data/example_input.png')
    outDir = os.path.join(rootDir, 'output/output.png')
    exampleOutDir = os.path.join(rootDir, 'data/example_output.png')
    print("input image: {}".format(inDir))
    print("output image: {}".format(outDir))

    setting = {
        'rgbLb': (250, 40, 110),
        'rgbUb': (255, 255, 210),
        'dilateErode': [2, -2],
        'alpha': 2.1,
        'beta': -127,
        'numCurves': 10
    }

    image = cv2.imread(inDir)
    cm = CurveMeasurer(setting)
    cm.detect(image)
    cm.show()
    cm.saveImage(outDir)
    
    exampleOut = cv2.imread(exampleOutDir)
    exampleOut = cv2.cvtColor(exampleOut, cv2.COLOR_BGR2RGB)
    assert( (cm.imgOut == exampleOut).all() )