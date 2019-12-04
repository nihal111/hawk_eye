import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

player_temps = []


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
    # plt.show()
    im2,contours,hierarchy = cv2.findContours(res_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    prev = 0
    x_all = []
    y_all = []
    h_all = []
    w_all = []
    font = cv2.FONT_HERSHEY_SIMPLEX
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
                #     #Mark blue jersy players as 
                # print(image.shape)
                # cv2.putText(image, 'Player', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                # cv2.rectangle(image,(x,y),(x+h,y+w),(255,0,0),thickness = -1)
                
                x_all.append(x)
                y_all.append(y)
                w_all.append(w)
                h_all.append(h)
                # print(x, h, y, w)
                if h > 0 and w > 0:
                    player_crop = image[y:y+h, x:x+w]
                    player_crop = cv2.resize(player_crop, (30, 30))
                #     plt.imshow(player_crop)
                #     plt.show()
                #     # print(player_crop.shape)
                    player_crop = player_crop.reshape(-1)
                    player_temps.append(player_crop)
                    # print(player_crop.shape)

                    
                # else:
                #     pass
                # if(nzCountred>=20):
                #     #Mark red jersy players as belgium
                #     cv2.putText(image, 'Belgium', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
                #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
                # else:
                    # pass
        # if((h>=1 and w>=1) and (h<=30 and w<=30)):
        #     player_img = image[y:y+h,x:x+w]
        
        #     player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
        #     #white ball  detection
        #     mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
        #     res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
        #     res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
        #     res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
        #     nzCount = cv2.countNonZero(res1)


        #     if(nzCount >= 3):
        #         # detect football
        #         cv2.putText(image, 'football', (x-2, y-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
        #         cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    player_temps_arr = np.array(player_temps)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(player_temps)
    plt.imshow(image)
    plt.show()
    labels = kmeans.labels_
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    w_all = np.array(w_all)
    h_all = np.array(h_all)
    coords = np.column_stack((x_all+h_all//2,y_all+w_all/2))
    for i in range(len(labels)):
        if(labels[i]==0):
            cv2.rectangle(image,(x_all[i],y_all[i]),(x_all[i]+h_all[i],y_all[i]+w_all[i]),(255,0,0),thickness = -1)
        elif(labels[i]==1):
            cv2.rectangle(image,(x_all[i],y_all[i]),(x_all[i]+h_all[i],y_all[i]+w_all[i]),(0,0,255),thickness = -1)
    # plt.scatter(x_all[0],y_all[0],color='red')
    # plt.scatter(x_all[0],y_all[0]+w_all[0],color='blue')
    # plt.scatter(x_all[0]+h_all[0],y_all[0]+w_all[0],color='yellow')
    # plt.scatter(coords[:,0],coords[:,1],color='white')
    # plt.scatter(x_all[0]+h_all[0],y_all[0],color='purple')
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
    return coords, labels

if __name__ == '__main__':
    
    for k in range(22, 23):
    
        file_name = 'soccer_data/test/' + str(k)
        football_field = 'football_field.jpg'

        # with open('{}.homographyMatrix'.format(file_name)) as f:
        #     content = f.readlines()
        # H = np.zeros((3, 3))
        # for i in range(len(content)):
        #     H[i] = np.array([float(x) for x in content[i].strip().split()])
        bgr = cv2.imread('{}.jpg'.format(file_name)).astype(np.uint8)
        inputIm = bgr[..., ::-1]

        coords, labels = frame_detect(inputIm)
        labels = np.array(labels)
        np.save('soccer_data/coords/'+str(k),coords)
        np.save('soccer_data/labels/'+str(k),labels)

           
        
        ## Change directory to appropriate perturbation
        # cv2.imwrite('trainA_pan/' + str(k)  + '.jpg', inputIm)

        # football = cv2.imread(football_field).astype(np.uint8)
        # footballIm = football[..., ::-1]

        # # plt.imshow(footballIm)
        # # plt.show()
        # warpIm = warpImage( bgr, footballIm, H, padding=200, idx = k)

        