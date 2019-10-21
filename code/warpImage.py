import numpy as np


def warpImage(inputIm, refIm, H, overwrite1=False):
    h, w, _ = inputIm.shape
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1],
                        [w - 1, h - 1, 1]]).transpose()
    transformed_corners = np.matmul(H, corners)
    transformed_corners = np.divide(
        transformed_corners, transformed_corners[2, :])

    x_min = min(0, min(transformed_corners[0, :]))
    x_max = max(refIm.shape[1], max(transformed_corners[0, :]))
    y_min = min(0, min(transformed_corners[1, :]))
    y_max = max(refIm.shape[0], max(transformed_corners[1, :]))

    x_width = int(x_max - x_min + 1)
    y_width = int(y_max - y_min + 1)

    warpIm = np.zeros((y_width, x_width, 3))
    mergeIm = np.zeros((y_width, x_width, 3))

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
        return x > 0 and x < inputIm.shape[1] and \
            y > 0 and y < inputIm.shape[0]

    for k in range(0, canvas_coords.shape[1]):
        x_canvas = int(canvas_coords[0, k] - x_min)
        y_canvas = int(canvas_coords[1, k] - y_min)
        x_input, y_input = int(inputIm_coords[0, k]), int(inputIm_coords[1, k])
        if inside_input(x_input, y_input):
            # Copy input image to refIm
            warpIm[y_canvas, x_canvas] = inputIm[y_input, x_input]
            mergeIm[y_canvas, x_canvas] = inputIm[y_input, x_input]

    for x in range(0, refIm.shape[1]):
        for y in range(0, refIm.shape[0]):
            x_canvas, y_canvas = int(x - x_min), int(y - y_min)
            mergeIm[y_canvas, x_canvas] = refIm[y, x]

    if(overwrite1):
        for k in range(0, canvas_coords.shape[1]):
            x_canvas = int(canvas_coords[0, k] - x_min)
            y_canvas = int(canvas_coords[1, k] - y_min)
            x_input, y_input = int(inputIm_coords[0, k]), int(
                inputIm_coords[1, k])
            if inside_input(x_input, y_input):
                # Copy input image to refIm
                warpIm[y_canvas, x_canvas] = inputIm[y_input, x_input]
                mergeIm[y_canvas, x_canvas] = inputIm[y_input, x_input]

    # Reshape warpIm to be tight bounding box
    if min(transformed_corners[0, :]) > 0 and \
            max(transformed_corners[0, :]) < refIm.shape[1]:
        warpIm = warpIm[:, int(min(transformed_corners[0, :])):int(
            max(transformed_corners[0, :])), :]
    elif min(transformed_corners[0, :]) > 0:
        warpIm = warpIm[:, int(min(transformed_corners[0, :])):, :]
    elif max(transformed_corners[0, :]) < refIm.shape[1]:
        warpIm = warpIm[:, :int(max(transformed_corners[0, :])), :]
    if min(transformed_corners[1, :]) > 0 and \
            max(transformed_corners[1, :]) < refIm.shape[0]:
        warpIm = warpIm[int(min(transformed_corners[1, :])):int(
            max(transformed_corners[1, :])), :, :]
    elif min(transformed_corners[1, :]) > 0:
        warpIm = warpIm[int(min(transformed_corners[1, :])):, :, :]
    elif max(transformed_corners[1, :]) < refIm.shape[0]:
        warpIm = warpIm[:int(max(transformed_corners[1, :])), :, :]

    return warpIm.astype('uint8'), mergeIm.astype('uint8')
