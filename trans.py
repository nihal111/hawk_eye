import numpy as np
import cv2
import matplotlib.pyplot as plt
from warpImage import warpImage

bgr=cv2.imread("24.jpg").astype(np.uint8)
im1=bgr[...,::-1]
H=([ [     6.3354158e-02  , 1.7767637e-01 , -4.3416524e+01],
  [-2.0370987e-02 ,  4.2005604e-01 , -4.8996014e+01],
  [-2.4303074e-05  , 3.1014795e-03  , 1.0000000e+00]])

warp=warpImage(im1,H)
plt.imshow(warp)
plt.show()
plt.clf()