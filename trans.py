import numpy as np
import cv2
import matplotlib.pyplot as plt
from computeH import computeH
from warpImage import warpImage

bgr=cv2.imread("8.jpg").astype(np.uint8)
im1=bgr[...,::-1]
H=([[ 9.9334547e-02  , 4.9523232e-01 ,  1.6050185e+02],
  [ 5.3051280e-02 ,  9.3862227e-01 , -2.1331082e+02],
  [-2.8968468e-05 ,  8.1729231e-03  , 1.0000000e+00]])

warp=warpImage(im1,H)
plt.imshow(warp)
plt.show()
plt.clf()