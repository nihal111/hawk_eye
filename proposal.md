---
layout: page
title: Proposal
---

# Hawk Eye
### Automatic Birds Eye View Registration of Sports Videos

## Motivation:
The sports broadcasting viewing experience has remained essentially unchanged for decades. A camera position on the side of the playing field which pans according to where the focus of the game is at that moment. However, with current technology and computer vision, we can overhaul this viewing experience from a purely 2D perspective and closer to a 3D experience. This would closely simulate the experience of attending a live sports event but without actually being physically present there. This transformed perspective could enable a person to make an individual player their focus even if the camera isn’t focusing on that person. 

## Method:


1. Obtaining a top view projection of the training image:
Annotate around 200 images with 4 point correspondences from the actual camera view image to the top-view projection blueprint.
Using these 4 correspondences we will generate the Homography matrix for these annotated images and their corresponding top-view.
We will use the blueprint to get the edge map using the inverse homography matrix.
We will now vary the pan, tilt and zoom of the camera angle by small margins which can be domain specific. For each of these variations we will generate the inverse homography matrix so we can generate the edge maps for each of these views.
We will compile these edge maps along with their corresponding top views into a dictionary which will be used later for querying.

2. Generating edge maps:
For images for which we don’t have an edge map, we will train a Pix2Pix network.
The input to the network will be the actual camera view images and the output will be the corresponding edge map. The training data we will be using is the 200 images used to obtain top views along with the edge maps generated from the inverse homography matrix. 

3. Obtaining top view projecting during testing:
The unseen actual camera view image will be first sent to the pix2pix network which generates the edge map for the image.
The edge map will be sent as a query to the database which will give us back to the most similar edge map using nearest neighbour approaches.
This retrieved edge map will give the corresponding homography matrix which can be used to obtain the top view for the actual camera image.

4. Player Detection:
Use DBScan to obtain the approximate location of a player. The clustering will be based on the colour and the size of the cluster will be tuned to ensure that 2 players do not get put in the same cluster.
Use the obtained cluster centers to obtain bounding boxes for each player.
The coordinates of these bounding boxes can be projected to the top view space by using the previously calculated homography matrices for each image. This will give us the location of each player in the top-view.


![placeholder]({{site.baseurl}}/public/camera_view.jpg "Camera View")
<center>a) Camera View from broadcast</center>  
<br/>

![placeholder]({{site.baseurl}}/public/top_view.png "Top View")
<center>b) Top View with camera region highlighted</center>

## Scope and Expected Results:

### 100% :
Input will be a camera broadcast image of a sports game.
This input will be transformed to the top view perspective.
We would be able to transform landmarks in broadcast images to the top view perspective, and this can be extended for doing it to a video.

### 125%
The top view perspective obtained can be used to for some analytics. The analytics could include event detection and player locations.
Video smoothing and stabilization could be done to make sure the trajectories of players in top view is good.
Train Faster RCNNs for a sport where labelled detections of players are there. Project players onto top view and use existing tracking algorithms like (Tracktor [https://github.com/phil-bergmann/tracking\_wo\_bnw](https://github.com/phil-bergmann/tracking_wo_bnw) or KLT based).

## Resources:
### Datasets
FIFA 2014 World Cup Dataset: [http://www.cs.toronto.edu/~namdar/data/soccer_data.tar.gz ](http://www.cs.toronto.edu/~namdar/data/soccer_data.tar.gz )  
Volleyball dataset: [https://github.com/mostafa-saad/deep-activity-rec ](https://github.com/mostafa-saad/deep-activity-rec )  
Pix2pix github codebase: [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)


### References:
[https://faculty.iiit.ac.in/~vgandhi/papers/sbgj_wacv_2018.pdf](https://faculty.iiit.ac.in/~vgandhi/papers/sbgj_wacv_2018.pdf)  
[http://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Chen_Sports_Camera_Calibration_via_Synthetic_Data_CVPRW_2019_paper.pdf](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Chen_Sports_Camera_Calibration_via_Synthetic_Data_CVPRW_2019_paper.pdf)  
[http://www.cs.toronto.edu/~namdar/pdfs/sports_cvpr_2017.pdf](http://www.cs.toronto.edu/~namdar/pdfs/sports_cvpr_2017.pdf)  
[https://www.cs.ubc.ca/~murphyk/Papers/weilwun-pami12.pdf](https://www.cs.ubc.ca/~murphyk/Papers/weilwun-pami12.pdf)  
[http://web.engr.oregonstate.edu/~afern/papers/register-cvpr07.pdf](http://web.engr.oregonstate.edu/~afern/papers/register-cvpr07.pdf)  
[https://www.cs.ubc.ca/~lowe/papers/okuma04b.pdf](https://www.cs.ubc.ca/~lowe/papers/okuma04b.pdf)

