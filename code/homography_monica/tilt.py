import numpy as np
import matplotlib.pyplot as plt


def tilt(points,t):
	ones = np.ones(points.shape[0])
	output=np.zeros(points.shape)
	output[0][0]=(1-t)*points[0][0]+t*points[3][0]
	output[0][1]=(1-t)*points[0][1]+t*points[3][1]

	output[3][0]=(-t)*points[0][0]+(t+1)*points[3][0]
	output[3][1]=(-t)*points[0][1]+(t+1)*points[3][1]

	output[1][0]=(1-t)*points[1][0]+t*points[2][0]
	output[1][1]=(1-t)*points[1][1]+t*points[2][1]

	output[2][0]=(-t)*points[1][0]+(t+1)*points[2][0]
	output[2][1]=(-t)*points[1][1]+(t+1)*points[2][1]
	return output




