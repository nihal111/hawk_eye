import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

cluster_centers = np.load('blue_or_red.npy')

def frame_detect(image):


    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #green range
    lower_green = np.array([30,100, 100])
    upper_green = np.array([90, 255, 255])
    #blue range
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    #Red range
    lower_red = np.array([0,31,255])
    upper_red = np.array([176,255,255])

    #white range
    lower_white = np.array([0,0,0])
    upper_white = np.array([0,0,255])

    #Define a mask ranging from lower to uppper
    mask = cv2.inRange(hsv, lower_green, upper_green)
    #Do masking
    res = cv2.bitwise_and(image, image, mask=mask)
    #convert to hsv to gray
    res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    #Defining a kernel to do morphological operation in threshold image to 
    #get better output.
    kernel = np.ones((13,13),np.uint8)
    thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    plt.imshow(image)
    plt.show()
    plt.imshow(res_gray)
    plt.show()

    im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    prev = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        #Detect players
        if(h>=1.2*(w)):
            if( w>10 ):
                player_img = image[y:y+h,x:x+w]
                player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
                #If player has blue jersy
                mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)
                #If player has red jersy
                mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
                nzCountred = cv2.countNonZero(res2)
                image = np.copy(image)

                # if(nzCount >= 20):
                #     #Mark blue jersy players as france
                
                # print("hello")
                # player_crop = image[y:y+h, x:x+w]
                # plt.imshow(player_crop)
                # plt.show()
                # player_crop = cv2.resize(player_crop, (30, 30))
                # player_crop = player_crop.reshape(-1)
                
                # dists = np.sum(np.abs(player_crop - cluster_centers), 1)
                # print(dists)
                # print(np.argmin(dists))
                
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),thickness = 3)
                # else:
            #     pass
            # if(nzCountred>=20):
            #     #Mark red jersy players as belgium
            #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
            # else:
                    # pass


    plt.imshow(image)
    plt.show()



if __name__ == '__main__':
    
    for k in range(10, 13):
    
        file_name = '/home/rohit/Documents/soccer_data/raw/train_val/' + str(k)

        bgr = cv2.imread('{}.jpg'.format(file_name)).astype(np.uint8)
        inputIm = bgr[..., ::-1]

        frame_detect(inputIm)

           

        