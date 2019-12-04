import numpy as np
import cv2
from matplotlib import path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.ops import split

#flip this if you want to visualize
visualize = False
only_score = False

def get_bounds(transformed_corners, footballIm, padding):
    x_min = min(0, min(transformed_corners[0, :]))
    x_max = max(footballIm.shape[1], max(transformed_corners[0, :]))
    y_min = min(0, min(transformed_corners[1, :]))
    y_max = max(footballIm.shape[0], max(transformed_corners[1, :]))

    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    return x_min, x_max, y_min, y_max


def warpImageOntoCanvas(inputIm, footballIm, H, x_min, x_max, y_min, y_max, top_left):
    x_width = int(x_max - x_min + 1)
    y_width = int(y_max - y_min + 1)
    canvasIm = np.zeros((y_width, x_width, 3))
    
    xs, ys, a = [], [], np.zeros((x_width, y_width))
    for index, _ in np.ndenumerate(a):
        xs.append(x_min + index[0]), ys.append(y_min + index[1])
    canvas_coords = np.vstack(
        (np.array(xs), np.array(ys), np.ones(len(xs))))
    transformed = np.matmul(np.linalg.inv(H), canvas_coords)
    xs = np.divide(transformed[0, :], transformed[2, :])
    ys = np.divide(transformed[1, :], transformed[2, :])
    inputIm_coords = np.transpose(np.column_stack((xs, ys)))

    def inside_input(x, y):
        return x >= 0 and x < inputIm.shape[1] and \
            y >= 0 and y < inputIm.shape[0]

    footballIm_h, footballIm_w, _ = footballIm.shape
    for i in range(footballIm_w):
        for j in range(footballIm_h):
            canvasIm[j][i] = footballIm[j][i]

    # Uncomment if you want to plot the warped image on the canvas

    for k in range(0, canvas_coords.shape[1]):
        x_canvas = int(canvas_coords[0, k] - x_min)
        y_canvas = int(canvas_coords[1, k] - y_min)
        x_input, y_input = int(inputIm_coords[0, k]), int(inputIm_coords[1, k])
        if inside_input(x_input, y_input):
            # Copy input image to refIm
            canvasIm[y_canvas, x_canvas] = inputIm[y_input, x_input]

    return np.array(canvasIm, dtype=int)


def transformAndShow(file_name, H, padding, top_left):
    bgr = cv2.imread(file_name).astype(np.uint8)
    inputIm = bgr[..., ::-1]

    # Change directory to appropriate perturbation
    # cv2.imwrite('trainA_pan/' + str(k)  + '.jpg', inputIm)

    football_field = 'football_field.jpg'
    football = cv2.imread(football_field).astype(np.uint8)
    footballIm = football[..., ::-1]

    # plt.imshow(footballIm)
    # plt.show()

    h, w, _ = bgr.shape
    # Find input image corners
    corners = np.array([[0, 0, 1], [w - 1, 0, 1],
                        [w - 1, h - 1, 1], [0, h - 1, 1]]).transpose()
    # Find trapezium corners in the warped space (top-view)
    transformed_corners = np.matmul(H, corners)
    transformed_corners = np.divide(
        transformed_corners, transformed_corners[2, :])
    
    # if only_score:
    #     return transformed_corners
    #     print("shouldnt be here")
    
    if top_left is None:
        x1, _, y1, _ = get_bounds(transformed_corners, footballIm, padding)
        top_left=(-x1, -y1)

    f_y, f_x, f_z = footballIm.shape
    new_footballIm = np.zeros((int(top_left[1]) + f_y, int(top_left[0]) + f_x, f_z))
    new_footballIm[int(top_left[1]):, int(top_left[0]):, :] = footballIm
    
    # plt.imshow(new_footballIm)
    # plt.show()
    
    # Get bounds with football field added and padding
    x_min, x_max, y_min, y_max = get_bounds(
        transformed_corners, new_footballIm, padding)

    # Get canvas with warped input and football field
    canvasIm=warpImageOntoCanvas(
        inputIm, new_footballIm, H, x_min, x_max, y_min, y_max, top_left)

    plt.cla()
    plt.clf()
    plt.imshow(canvasIm)
    plt.show()
    fig = plt.gcf()
    ax = fig.gca()

    points = [(corner[0] - x_min, corner[1] - y_min) for corner in transformed_corners.T]
    print(points)
    circle1 = plt.Circle(points[0], 4, color='r')
    circle2 = plt.Circle(points[1], 4, color='g')
    circle3 = plt.Circle(points[2], 4, color='b')
    circle4 = plt.Circle(points[3], 4, color='pink')
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    trapezium = Polygon(points)
    all_points = points
    
    points = [(0,0), (115, 0), (115, 75), (0, 75)]
    points = [(corner[0] + top_left[0], corner[1] + top_left[1]) for corner in points]
    points.append(points[0])
    circle1 = plt.Circle(points[0], 4, color='cyan')
    circle2 = plt.Circle(points[1], 4, color='cyan')
    circle3 = plt.Circle(points[2], 4, color='cyan')
    circle4 = plt.Circle(points[3], 4, color='cyan')
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    if visualize:
        plt.show()
    football_field = Polygon(points)
    all_points += points

    plt.plot(*football_field.exterior.xy)
    plt.plot(*trapezium.exterior.xy)
    
    if visualize:
        plt.show()

    lines = []
    x_min1 = min([p[0] for p in all_points])
    x_max1 = max([p[0] for p in all_points])
    y_min1 = min([p[1] for p in all_points])
    y_max1 = max([p[1] for p in all_points])
    for i in range(0, 4):
        p1, p2 = list(points[i]), list(points[i+1])
        if p1[0] != p2[0]:
            if p1[0] < p2[0]:
                p1[0], p2[0] = x_min1, x_max1
            else:
                p1[0], p2[0] = x_max1, x_min1

        if p1[1] != p2[1]:
            if p1[1] < p2[1]:
                p1[1], p2[1] = y_min1, y_max1
            else:
                p1[1], p2[1] = y_max1, y_min1
        lines.append(LineString([tuple(p1), tuple(p2)]))

    # Find the interior polygon
    line = lines[0] # p0 --- p1
    splitted = split(trapezium, line)
    if len(splitted) == 2:
        a, b = splitted
        y_avg = sum(a.exterior.coords.xy[1]) / len(a.exterior.coords.xy[1])
        # Choose polygon above line
        trapezium = a if y_avg > points[1][1] else b
    plt.plot(*football_field.exterior.xy)
    plt.plot(*trapezium.exterior.xy)
    x, y = line.xy
    plt.plot(x, y, 'o', color='#999999', zorder=1)
    
    if visualize:
        plt.show()

    line = lines[1] # p1 --- p2
    splitted = split(trapezium, line)
    if len(splitted) == 2:
        a, b = splitted
        x_avg = sum(a.exterior.coords.xy[0]) / len(a.exterior.coords.xy[0])
        # Choose polygon left of line
        trapezium = a if x_avg < points[2][0] else b
    plt.plot(*football_field.exterior.xy)
    plt.plot(*trapezium.exterior.xy)
    x, y = line.xy
    plt.plot(x, y, 'o', color='#999999', zorder=1)
    
    if visualize:
        plt.show()

    line = lines[2] # p2 --- p3
    splitted = split(trapezium, line)
    if len(splitted) == 2:
        a, b = splitted
        y_avg = sum(a.exterior.coords.xy[1]) / len(a.exterior.coords.xy[1])
        # Choose polygon below line
        trapezium = a if y_avg < points[3][1] else b
    plt.plot(*football_field.exterior.xy)
    plt.plot(*trapezium.exterior.xy)
    x, y = line.xy
    plt.plot(x, y, 'o', color='#999999', zorder=1)
    
    if visualize:
        plt.show()

    line = lines[3] # p3 --- p1
    splitted = split(trapezium, line)
    if len(splitted) == 2:
        a, b = splitted
        x_avg = sum(a.exterior.coords.xy[0]) / len(a.exterior.coords.xy[0])
        # Choose polygon right of line
        trapezium = a if x_avg > points[3][0] else b
    plt.plot(*football_field.exterior.xy)
    plt.plot(*trapezium.exterior.xy)
    x, y = line.xy
    plt.plot(x, y, 'o', color='#999999', zorder=1)
    
    if visualize:
        plt.show()

    # print("Polygon formed")
    # print(trapezium)
    cut_polygon_corners = [[a + x_min, b + y_min] for a, b in zip(trapezium.exterior.coords.xy[0], trapezium.exterior.coords.xy[1])]
    # print(cut_polygon_corners)
    return cut_polygon_corners


if __name__ == '__main__':

    # For dictionary images, the homographies we have saved, transforms the
    # camera image into the top view space where football field's top left
    # corner is at `top_left` (xmin, ymin)

    # For test images from the dataset, the homography transforms the camera
    # image into the top view space with the top left corner of the football
    # field is at (0, 0)

    # Displaying an example of each:
    # Flip visualize to True near imports if you want to show the outputs
    # make only_score to False if you want to use normally

    # ----- Image from dictionary -----
    
    file_name = 'soccer_data/train_zoom/148_85.jpg'
    homography_file = 'soccer_data/train_zoom/H148_85.npy'
    H = np.load(homography_file)
    
    with open('soccer_data/top_left/148.txt') as f:
        content = [float(line.strip()) for line in f.readlines()]
    top_left = (content[0], content[1])
    
    # print(file_name, H, 0, top_left)
    
    transformed_corners = transformAndShow(file_name, H, padding=0, top_left=top_left)
    transformed_corners_dict = [[corner[0] - top_left[0], corner[1] - top_left[1]] 
                            for corner in transformed_corners]
    print("Trapezium corners from dictionary image-\n", transformed_corners_dict)

    # Football field coordinates are- (0, 0), (115, 0), (115, 75), (0, 75)


    # ------ Image from Dataset ------

    file_name = 'soccer_data/train_val/148.jpg'
    homography_file = 'soccer_data/train_val/148.homographyMatrix'
    
    with open(homography_file) as f:
        content = f.readlines()
        
    H = np.zeros((3, 3))
    for i in range(len(content)):
        H[i] = np.array([float(x) for x in content[i].strip().split()])
    top_left = None
    
    # print(file_name, H, 0, top_left)
    
    transformed_corners = transformAndShow(file_name, H, padding=0, top_left=top_left)
    transformed_corners_database = [[corner[0], corner[1]] 
                            for corner in transformed_corners]
    
    print("\nTrapezium corners from dataset image-\n", transformed_corners_database)
    
    
    
    a = Polygon([(x[0], x[1]) for x in transformed_corners_dict])
    b = Polygon([(x[0], x[1]) for x in transformed_corners_database])
    plt.plot(*a.exterior.xy)
    plt.plot(*b.exterior.xy)
    
    if visualize:
        plt.show()

    IoU = a.intersection(b).area / a.union(b).area
    
    print("IoU: ", IoU)