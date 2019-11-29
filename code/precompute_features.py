import os 
import os.path as op
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

#descriptors
descriptors = ['hog', 'distance_transform']

hog = cv2.HOGDescriptor()

database_path = '/home/rohit/Documents/Sem1_Gatech/cv/cv_project/train_augmented_data'
database_folders = ['pan', 'tilt', 'zoom', 'normal']

database_files = []
for folder in database_folders:
    dpath = op.join(database_path, 'train_' + folder)
    files = os.listdir(dpath)
    files = [op.join(dpath, file) for file in files if file[-4:] == '.png' or file[-4:] == '.jpg']
    database_files.append(files)

database_files = sum(database_files, [])
database_hogs = []
database_dtrans = []

for i, train_file in enumerate(database_files):
    print(train_file)
    
    im_gray = cv2.imread(train_file, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.resize(im_gray, (256, 256))
    thresh = 127
    train_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    
    train_feat_dtrans = ndimage.distance_transform_edt(train_bw)
    train_feat_hog = hog.compute(train_bw)
    
    database_hogs.append(train_feat_hog)
    database_dtrans.append(train_feat_dtrans)
    
database_hogs = np.array(database_hogs)
database_dtrans = np.array(database_dtrans)

np.save('database_hogs.npy', database_hogs)
np.save('database_dtrans.npy', database_dtrans)