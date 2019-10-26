import numpy as np
import cv2
from matplotlib import path
import matplotlib.pyplot as plt
from pan import pan
from computeH import computeH


def getMask(points, x_width, y_width):
    # points are in the order TL, TR, BR, BL
    mask = np.zeros((x_width, y_width))
    p = path.Path([list(cood) for cood in points])
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if p.contains_point([i, j]):
                mask[i, j] = 1
    mask = mask.T
    return mask


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


def apply_perturbation(shifted_corners, perturbation='PAN'):
    if perturbation == 'PAN':
        pan_points = np.array(pan(shifted_corners, 0.2))
        return pan_points


def warpImageOntoCanvas(inputIm, footballIm, H, x_min, x_max, y_min, y_max):
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

    for k in range(0, canvas_coords.shape[1]):
        x_canvas = int(canvas_coords[0, k] - x_min)
        y_canvas = int(canvas_coords[1, k] - y_min)
        x_input, y_input = int(inputIm_coords[0, k]), int(inputIm_coords[1, k])
        if inside_input(x_input, y_input):
            # Copy input image to refIm
            canvasIm[y_canvas, x_canvas] = inputIm[y_input, x_input]
    footballIm_h, footballIm_w, _ = footballIm.shape
    for i in range(footballIm_w):
        for j in range(footballIm_h):
            canvasIm[j - int(y_min)][i - int(x_min)] = footballIm[j][i]

    return canvasIm


def perturbedToRect(inputIm, canvasIm, H):
    xs, ys, a = [], [], np.zeros((inputIm.shape[1], inputIm.shape[0]))
    for index, _ in np.ndenumerate(a):
        xs.append(index[0]), ys.append(index[1])
    input_coords = np.vstack(
        (np.array(xs), np.array(ys), np.ones(len(xs))))
    transformed = np.matmul(H, input_coords)
    transformed[0, :] = np.divide(transformed[0, :], transformed[2, :])
    transformed[1, :] = np.divide(transformed[1, :], transformed[2, :])

    edge_map_perturb = np.zeros(inputIm.shape)
    for k in range(0, input_coords.shape[1]):
        x_input = int(input_coords[0, k])
        y_input = int(input_coords[1, k])
        x_canvas = int(transformed[0, k])
        y_canvas = int(transformed[1, k])
        edge_map_perturb[y_input, x_input] = canvasIm[y_canvas, x_canvas]
    return edge_map_perturb


def warpImage(inputIm, footballIm, H, padding):
    global fig
    h, w, _ = inputIm.shape
    corners = np.array([[0, 0, 1], [w - 1, 0, 1],
                        [w - 1, h - 1, 1], [0, h - 1, 1]]).transpose()
    transformed_corners = np.matmul(H, corners)
    transformed_corners = np.divide(
        transformed_corners, transformed_corners[2, :])

    x_min, x_max, y_min, y_max = get_bounds(
        transformed_corners, footballIm, padding)

    shifted_corners = np.array([[corner[0] - x_min, corner[1] - y_min]
                                for corner in transformed_corners.T])
    pan_points = apply_perturbation(shifted_corners)

    non_homo_corners = np.array([[corner[0], corner[1]]
                                 for corner in corners.T])
    H_perturb = cv2.findHomography(non_homo_corners, pan_points)[0]
    H_perturb = computeH(non_homo_corners.T, pan_points.T)

    x_width = int(x_max - x_min + 1)
    y_width = int(y_max - y_min + 1)

    canvasIm = warpImageOntoCanvas(
        inputIm, footballIm, H, x_min, x_max, y_min, y_max)
    mask = getMask(pan_points, x_width, y_width)

    edge_map_perturb = perturbedToRect(inputIm, canvasIm, H_perturb)

    return canvasIm.astype('uint8'), pan_points, mask, \
        edge_map_perturb.astype('uint8')


def cv2warp(inputIm, H):
    w = 75
    h = 75
    warpIm = np.zeros((h, w, 3), "uint8")
    cv2.warpPerspective(
        src=inputIm, dst=warpIm, M=H, dsize=(h, w))


if __name__ == '__main__':
    file_name = 'soccer_data/train_val/26'
    football_field = 'football_field.jpg'

    with open('{}.homographyMatrix'.format(file_name)) as f:
        content = f.readlines()
    H = np.zeros((3, 3))
    for i in range(len(content)):
        H[i] = np.array([float(x) for x in content[i].strip().split()])
    bgr = cv2.imread('{}.jpg'.format(file_name)).astype(np.uint8)
    inputIm = bgr[..., ::-1]

    football = cv2.imread(football_field).astype(np.uint8)
    footballIm = football[..., ::-1]

    plt.imshow(footballIm)
    plt.show()
    warpIm, pan_points, mask, edge_map_perturb = warpImage(
        bgr, footballIm, H, padding=200)

    fig = plt.figure()
    plt.imshow(warpIm)
    for (x, y) in pan_points:
        circle = plt.Circle((x, y), 2, color=(1, 0, 0), fill=True)
        fig.add_subplot().add_artist(circle)
    plt.show()
    plt.imshow(mask)
    plt.show()
    plt.imshow(edge_map_perturb)
    plt.show()
