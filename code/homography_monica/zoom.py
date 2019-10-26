import matplotlib.pyplot as plt
import numpy as np

def zoom(points, zoomx, zoomy):
	ones = np.ones(points.shape[0])
	h1 = np.column_stack((points, ones.T))
	cx = np.mean(points[:, 0])
	cy = np.mean(points[:, 1])
	H = np.array([[zoomx, 0, 0], [0, zoomy, 0], [0, 0, 1]])
	I = np.array([[1, 0 ,-cx], [0, 1, -cy], [0, 0, 1]])
	I2 = np.array([[1, 0 ,cx], [0, 1, cy], [0, 0, 1]])
	G = np.matmul(H , I)
	outputs = np.matmul(I2,np.matmul(G, h1.T))[:-1,:].T
	return outputs


