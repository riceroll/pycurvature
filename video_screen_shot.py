import cv2
import tqdm

cap = cv2.VideoCapture('/Users/Roll/Desktop/bending_samples/1000.MOV')

# Get the frames per second
fps = round(cap.get(cv2.CAP_PROP_FPS))

# Get the total numer of frames in the video.
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frame_number = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number) # optional
success, image = cap.read()

step = 20

for i in tqdm.tqdm(range(int(60 / step * 4 ))):
    frame_number += 30 * 20
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = cap.read()
    image = image[:int(image.shape[0] / 2), :, :]
    cv2.imwrite('/Users/Roll/Desktop/bending_samples/1000.MOV' + "_" + str(frame_number) + ".png", image)
