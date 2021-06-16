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

def getDefaultSetting():
    from pycurve.pycurve import CurveMeasurer
    return CurveMeasurer.getDefaultSetting()

def analyzeVideo(inDir,
                 t0=2,
                 dt=10.2,
                 setting=None,
                 outDir=None,
                 testing=False):
    """
    
    :param inDir: input video directory
    :param t0: the video time of the first frame to be processed (in second)
    :param dt: the time interval between two frames to be processed (in second)
    :param setting: setting for CV, modify on top of the default setting using ``setting = getDefaultSetting()``
    :param outDir:  output video directory
    :param testing: if True, testing mode is enabled and processing figures will be displayed, press 'q' to skip to the next
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

    if setting is None:
        setting = CurveMeasurer.getDefaultSetting()
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
            successful = cm.detect(image)
            if testing:
                cm.show()
            if not successful:
                cm.show()
                break
            cm.saveImage(os.path.join(outDir, '{:.2f}.png').format(t))
            assert(len(cm.rs) == 1)
            textItems.append("{}\t{:.2f}\t{:.2f}".format(iActuation, t, cm.rs[0]))
            t += dt
        else:
            break

    text = "\n".join(textItems)
    with open(os.path.join(outDir, 'data.txt'), 'w') as ofile:
        ofile.write(text)