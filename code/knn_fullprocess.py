import os 
import os.path as op
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

#descriptors
descriptors = ['hog', 'distance_transform']

hog = cv2.HOGDescriptor()

query_path = './pix2pix_code/results/rgb2edge/test_latest/images'
query_files = os.listdir(query_path)
query_files = [file for file in query_files if file[-8:] == 'fake.png'][:20]

#need to precompute features for database locally then query
database_path = './pix2pix_code/results/rgb2edge/test_latest/images'
database_files = os.listdir(database_path)
database_files = [file for file in database_files]


for descriptor in descriptors:
    
    plt.suptitle(descriptor)
    
    sample_path = op.join(query_path, query_files[5])
    im_sample_gray = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    thresh = 127
    sample_bw = cv2.threshold(im_sample_gray, thresh, 255, cv2.THRESH_BINARY)[1]

    plt.subplot(3, 7, 1)
    plt.title('query')
    plt.imshow(sample_bw, cmap='gray')

    for i, qfile in enumerate(query_files):
        
        q_path = op.join(query_path, qfile)
        im_gray = cv2.imread(q_path, cv2.IMREAD_GRAYSCALE)
        thresh = 127
        query_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        
        if descriptor == 'distance_transform':
            query_feat = ndimage.distance_transform_edt(query_bw)
            sample_feat = sample_bw
            
            dist = 1/((np.sum(sample_feat * query_feat)) / (np.linalg.norm(sample_feat, ord=1)))
            
        elif descriptor == 'hog':
            query_feat = hog.compute(query_bw)
            sample_feat = hog.compute(sample_bw)
        
            dist = np.sum(np.abs(query_feat - sample_feat))
            
        print(dist)
        
        plt.subplot(3, 7, i + 2)
        plt.imshow(query_bw, cmap='gray')
        plt.title(str(dist))
        
    plt.show()
