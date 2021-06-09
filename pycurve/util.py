import cv2
import tqdm

def videoCapture(dir, second):
    """
    capture the nth second frame of a video
    :param dir: directory of the video
    :param second: the frame at the beginning of which is to be captured
    :return: np.array image
    """
    cap = cv2.VideoCapture(dir)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    iFrame = int(second * fps)
    if iFrame >= frame_count:
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, iFrame)  # optional
    success, image = cap.read()
    return image

def getVideoTime(dir):
    """
    
    :param dir: video directory
    :return: time of the video in second
    """
    cap = cv2.VideoCapture(dir)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return frame_count / fps

def analyzeVideo(inDir, t0=2, dt=10.2 ,outDir=None):
    """
    
    :param inDir: input video directory
    :param t0: initial time (second)
    :param dt: time interval (second)
    :param outDir:  output video directory
    :return:
    """
    import os
    import argparse
    import tqdm
    from pycurve.pycurve import CurveMeasurer
    
    if outDir is None:
        basename = '.'.join(os.path.basename(inDir).split('.')[:-1])
        outDir = os.path.join('.', basename)
        if not os.path.exists(outDir):
            os.makedirs(outDir)

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

    t0 = t0
    dt = dt
    T = getVideoTime(inDir)

    t = t0
    for iActuation in tqdm.tqdm(range(int(T // dt + 1))):
        ret = videoCapture(inDir, t)
        if ret is not False:
            image = ret
            cm.detect(image)
            cm.saveImage(os.path.join(outDir, '{:.2f}.png').format(t))
            textItems.append("{}\t{:.2f}\t{:.2f}".format(iActuation, t, cm.rs[0]))
            # cm.show()
            t += dt
        else:
            break
    
        # break

    text = "\n".join(textItems)
    with open(os.path.join(outDir, 'data.txt'), 'w') as ofile:
        ofile.write(text)