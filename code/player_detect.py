import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def frame_detect(image):
    # vidcap = cv2.VideoCapture('cutvideo.mp4')
    # success,image = vidcap.read()
    idx = 0
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #green range
    lower_green = np.array([45,100, 100])
    upper_green = np.array([75, 255, 255])
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
    # plt.imshow(res_gray, cmap ='gray')
    # plt.show()

    #Defining a kernel to do morphological operation in threshold image to 
    #get better output.
    # kernel = np.ones((13,13),np.uint8)
    # thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(thresh, cmap ='gray')
    # plt.show()
    # thresh = cv2.resize(thresh, (256,256))
    # thresh = cv2.resize(thresh, (256,256))
    #find contours in threshold image     
    plt.imshow(res_gray)
    plt.show()
    im2,contours,hierarchy = cv2.findContours(res_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    prev = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    # print(len(contours))
    for c in contours:
        x,y,h,w = cv2.boundingRect(c)
        #Detect players
        if(w>=h):
            if(w>12 and h>= 12):
                # print(x,y,w,h)

                idx = idx+1
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
                cv2.putText(image, 'Player', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                cv2.rectangle(image,(x,y),(x+h,y+w),(255,0,0),thickness = -1)
                # else:
                #     pass
                # if(nzCountred>=20):
                #     #Mark red jersy players as belgium
                #     cv2.putText(image, 'Belgium', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
                #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
                # else:
                    # pass
        if((h>=1 and w>=1) and (h<=30 and w<=30)):
            player_img = image[y:y+h,x:x+w]
        
            player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
            #white ball  detection
            mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
            res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
            res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
            nzCount = cv2.countNonZero(res1)


            if(nzCount >= 3):
                # detect football
                cv2.putText(image, 'football', (x-2, y-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)

    plt.imshow(image)
    plt.show()
    #         if(nzCount >= 3):
    #             # detect football
    #             cv2.putText(frame, 'football', (x-2, y-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
    #             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    #     cv2.imwrite("./Cropped/frame%d.jpg" % count, res)
    #     print( 'Read a new frame: ', success  )   # save frame as JPEG file	
    #     count += 1
    #     cv2.imshow('Match Detection',frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
        
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    
    for k in range(5, 12):
    
        file_name = 'soccer_data/train_val/' + str(k)
        football_field = 'football_field.jpg'

        # with open('{}.homographyMatrix'.format(file_name)) as f:
        #     content = f.readlines()
        # H = np.zeros((3, 3))
        # for i in range(len(content)):
        #     H[i] = np.array([float(x) for x in content[i].strip().split()])
        bgr = cv2.imread('{}.jpg'.format(file_name)).astype(np.uint8)
        inputIm = bgr[..., ::-1]

        frame_detect(inputIm)

           
        
        ## Change directory to appropriate perturbation
        # cv2.imwrite('trainA_pan/' + str(k)  + '.jpg', inputIm)

        # football = cv2.imread(football_field).astype(np.uint8)
        # footballIm = football[..., ::-1]

        # # plt.imshow(footballIm)
        # # plt.show()
        # warpIm = warpImage( bgr, footballIm, H, padding=200, idx = k)

        