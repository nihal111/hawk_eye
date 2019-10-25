import numpy as np
import cv2
import matplotlib.pyplot as plt
from warpImage import warpImage

bgr = cv2.imread("8.jpg").astype(np.uint8)
inputIm = bgr[..., ::-1]
bgr = cv2.imread("2.jpg").astype(np.uint8)
inputIm2 = bgr[..., ::-1]
H = ([[9.9334547e-02, 4.9523232e-01, 1.6050185e+02],
      [5.3051280e-02, 9.3862227e-01, -2.1331082e+02],
      [-2.8968468e-05, 8.1729231e-03, 1.0000000e+00]])

H2 = ([[1.1472719e-01, 4.6624011e-01, -1.4405836e+02],
       [-4.0034748e-02, 9.4924495e-01, -1.9838942e+02],
       [5.2846773e-05, 8.6005821e-03, 1.0000000e+00]])

w = 150
h = 150
warpIm = np.zeros((h, w, 3), "uint8")
warpIm2 = np.zeros((h, w, 3), "uint8")
mergeIm = np.zeros((h, w, 3), "uint8")
cv2.warpPerspective(src=inputIm, dst=warpIm, M=np.array(H), dsize=(h, w))
cv2.warpPerspective(src=inputIm2, dst=warpIm2, M=np.array(H2), dsize=(h, w))
for i in range(w):
    for j in range(h):
        if np.any(warpIm[j][i] != 0):
            mergeIm[j][i] = warpIm[j][i]
        if np.any(warpIm2[j][i] != 0):
            mergeIm[j][i] = warpIm2[j][i]
plt.imshow(mergeIm)
plt.show()
plt.clf()
