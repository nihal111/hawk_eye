import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor

clicks_w1 = np.zeros([4,2])
it = 0
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    global it

    clicks_w1[it] = [event.xdata, event.ydata]
    it = it+1

image1 = plt.imread("soccer_data/raw/train_val/1.jpg")
im1 = image1.copy()
im1.astype('uint8')
fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
ax = fig.add_subplot(111, facecolor='#FFFFCC')
plt.imshow(im1)
cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
plt.show()
np.save("soccer_data/raw/grass_bounds/1", clicks_w1)






