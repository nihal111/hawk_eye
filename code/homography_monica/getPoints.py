def getPoints(width,height,H):	
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
	return points,point