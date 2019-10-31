import matplotlib.pyplot as plt
import numpy as np

def zoom(points,sx = 0.5, sy = 0.5):
    # points in order TL TR BR BL
    ones = np.ones(points.shape[0])
    homo_points = np.column_stack((points, ones.T))
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    H = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    I1 = np.array([[1, 0 ,-cx], [0, 1, -cy], [0, 0, 1]])
    I2 = np.array([[1, 0 ,cx], [0, 1, cy], [0, 0, 1]])
    zoom_center = np.matmul(H , I1)
    outputs = np.matmul(I2,np.matmul(zoom_center, homo_points.T))[:-1,:].T
    
    return outputs


