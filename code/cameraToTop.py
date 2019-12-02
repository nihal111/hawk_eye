import numpy as np
import cv2
from matplotlib import path
import matplotlib.pyplot as plt


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
    print(canvasIm.shape)
    
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
    print(top_left)
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

    plt.imshow(inputIm)
    plt.show()

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

    print("transformed_corners\n", transformed_corners)
    
    if top_left is None:
        x1, _, y1, _ = get_bounds(transformed_corners, footballIm, padding)
        top_left=(-x1, -y1)
        print("New top left", top_left)

    f_y, f_x, f_z = footballIm.shape
    new_footballIm = np.zeros((int(top_left[1]) + f_y, int(top_left[0]) + f_x, f_z))
    new_footballIm[int(top_left[1]):, int(top_left[0]):, :] = footballIm
    
    plt.imshow(new_footballIm)
    plt.show()
    
    # Get bounds with football field added and padding
    x_min, x_max, y_min, y_max = get_bounds(
        transformed_corners, new_footballIm, padding)

    print("Inside transformAndShow")
    print(x_min, x_max, y_min, y_max)

    # Get canvas with warped input and football field
    canvasIm=warpImageOntoCanvas(
        inputIm, new_footballIm, H, x_min, x_max, y_min, y_max, top_left)

    plt.imshow(canvasIm)
    plt.show()


if __name__ == '__main__':
    # Below code is wrong
    # file_name='soccer_data/train_zoom/2_110' + '.jpg'
    file_name = '/home/rohit/Documents/soccer_data/raw/train_val/' + str(57) + '.jpg'
    padding=0

    # homography_file='soccer_data/train_zoom/H2_110.npy'
    # H=np.load(homography_file)

    with open('/home/rohit/Documents/soccer_data/raw/train_val/' + str(57) + '.homographyMatrix') as f:
        content = f.readlines()
    H = np.zeros((3, 3))
    for i in range(len(content)):
        H[i] = np.array([float(x) for x in content[i].strip().split()])

    corners = np.array([[0, 0, 1]]).transpose()
    transformed_corners = np.matmul(H, corners)
    transformed_corners = np.divide(
        transformed_corners, transformed_corners[2, :])
    
    transformAndShow(file_name, H, padding, top_left=None)
