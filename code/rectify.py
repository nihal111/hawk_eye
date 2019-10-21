import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from warpImage import warpImage

file_name = 'soccer_data/train_val/2'

with open('{}.homographyMatrix'.format(file_name)) as f:
    content = f.readlines()

H = np.zeros((3, 3))

for i in range(len(content)):
    H[i] = np.array([float(x) for x in content[i].strip().split()])

inputIm = np.array(Image.open('{}.jpg'.format(file_name)))
refIm = np.array(np.zeros(inputIm.shape))


warpIm, mergeIm = warpImage(inputIm, refIm, H)
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].imshow(inputIm)
ax[0].set_title('inputIm')
ax[1].imshow(warpIm)
ax[1].set_title('warpIm')
plt.show()
