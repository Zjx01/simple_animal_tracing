from doctest import FAIL_FAST
from skimage import io
import cv2
import numpy as np
file_name = "black_1_short.mp4"
video = cv2.VideoCapture(file_name)
imgs = []
while True:
    ret, frame = video.read()
    if ret:
        imgs.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[np.newaxis,:,:])
    else:
        break
imgs = np.vstack(imgs)
io.imsave(file_name[:-4]+".tif",imgs,check_contrast=False)
