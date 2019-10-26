import numpy as np
import matplotlib.pyplot as plt


def tilt(points,t = 0.1):
	#points in order TL TR BR BL

	output = np.zeros(points.shape)

	output[0] = (1-t)*points[0] + t*points[3]
	output[3] = (-t)*points[0] + (t+1)*points[3]
	output[1] = (1-t)*points[1] + t*points[2]
	output[2] = (-t)*points[1] + (t+1)*points[2]

	return output




