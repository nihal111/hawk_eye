import numpy as np

def warpImage(inputIm, H):
	width=inputIm.shape[1]
	height=inputIm.shape[0]
	points,point=getPoints(w,h,H)
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
	return warpIm