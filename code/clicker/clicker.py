import numpy as np 
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor

global clicks_w , it


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    global clicks_w, it
    clicks_w[it] = [event.xdata, event.ydata]
    it = it+1

bound_dict = {}

### Change the ranges of the for loop for the images you've been assigned
for i in range(1,50):
    image_num = 1
    global clicks_w, it
    clicks_w = np.zeros([4,2])
    it = 0
    image = plt.imread("soccer_data/raw/train_val/"+str(i)+".jpg")
    im = image.copy()
    im.astype('uint8')
    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax = fig.add_subplot(111, facecolor='#FFFFCC')
    plt.imshow(im)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    plt.show()
    
    bound_dict[str(i)+".jpg"] = clicks_w
print(bound_dict)

### Change pickle file name to the range of images you've been assigned
f = open("soccer_data/raw/bounds_dict_1_50.pkl","wb")
pickle.dump(bound_dict,f)
f.close()









