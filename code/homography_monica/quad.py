import matplotlib.pyplot as plt
import numpy as np


points = np.array([[1, 5], [3, 5], [1, 1], [4, 1]])
ones = np.ones(points.shape[0])

h1=np.column_stack((points, ones.T))

cx = np.mean(points[:, 0])
cy = np.mean(points[:, 1])

sx = 2
sy = 2

H = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

I = np.array([[1, 0 ,-cx], [0, 1, -cy], [0, 0, 1]])
I2 = np.array([[1, 0 ,cx], [0, 1, cy], [0, 0, 1]])

G = np.matmul(H , I)

outputs = np.matmul(I2,np.matmul(G, h1.T))[:-1,:].T
print(outputs)


