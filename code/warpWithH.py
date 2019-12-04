import numpy as np
import cv2
import matplotlib.pyplot as plt
bgr = cv2.imread('./soccer_data/train_zoom/23_110.jpg')
inputIm = bgr[..., ::-1]
H_inv = np.load('./soccer_data/train_zoom/H23_110.npy') 
H = H_inv
width=inputIm.shape[1]
height=inputIm.shape[0]
points=np.zeros((4,2))
point=np.zeros(3)
point[2]=1
points[0]=(np.matmul(H,point.T)/np.matmul(H,point.T)[2])[:2]
point[1]=height-1
points[1]=(np.matmul(H,point.T)/np.matmul(H,point.T)[2])[:2]
point[0]=width-1
points[3]=(np.matmul(H,point.T)/np.matmul(H,point.T)[2])[:2]
point[1]=0
points[2]=(np.matmul(H,point.T)/np.matmul(H,point.T)[2])[:2]
min_xy=np.amin(points,axis=0)
max_xy=np.amax(points,axis=0)
w=int(max_xy[0]-min_xy[0])
h=int(max_xy[1]-min_xy[1])
warpIm=np.zeros((h,w,3),"uint8")
inv=np.linalg.inv(H)
for i in range(w):
	for j in range(h):
		point[0]=i+int(min_xy[0])
		point[1]=j+int(min_xy[1])
		coord=(np.matmul(inv,point.T)/np.matmul(inv,point.T)[2])[:2]
		in0=int(coord[0])
		in1=int(coord[1])
		if(in0>=0 and in0<width and in1<height and in1>=0):
			warpIm[j][i]=inputIm[in1][in0]
plt.imshow(warpIm)
plt.show()