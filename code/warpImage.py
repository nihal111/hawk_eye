import numpy as np
import cv2
import matplotlib.pyplot as plt


def warpImage(inputIm, footballIm, H, padding):
    h, w, _ = inputIm.shape
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1],
                        [w - 1, h - 1, 1]]).transpose()
    transformed_corners = np.matmul(H, corners)
    transformed_corners = np.divide(
        transformed_corners, transformed_corners[2, :])

    x_min = min(0, min(transformed_corners[0, :]))
    x_max = max(footballIm.shape[1], max(transformed_corners[0, :]))
    y_min = min(0, min(transformed_corners[1, :]))
    y_max = max(footballIm.shape[0], max(transformed_corners[1, :]))

    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    x_width = int(x_max - x_min + 1)
    y_width = int(y_max - y_min + 1)

    warpIm = np.zeros((y_width, x_width, 3))

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
            warpIm[y_canvas, x_canvas] = inputIm[y_input, x_input]
    footballIm_h, footballIm_w, _ = footballIm.shape
    for i in range(footballIm_w):
        for j in range(footballIm_h):
            warpIm[j - int(y_min)][i - int(x_min)] = footballIm[j][i]

    return warpIm.astype('uint8')


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
    warpIm = warpImage(bgr, footballIm, H, padding=200)

    plt.imshow(warpIm)
    plt.show()
