import numpy as np


def computeH(t1, t2):
    num_pts = t1.shape[1]
    assert t1.shape[0] == t2.shape[0] == 2
    assert t1.shape[1] == t2.shape[1] == num_pts

    # Convert to homogenous coordinates
    pts1 = np.vstack((t1, np.ones(t1.shape[1])))
    pts2 = np.vstack((t2, np.ones(t2.shape[1])))

    # Create skeleton for L (2*n, 9)
    L = np.zeros((2 * num_pts, 3 * 3))

    # Populate L
    for i in range(num_pts):
        pt_T = pts1[:, i]
        ui_pt_T = pts2[0, i] * pt_T
        vi_pt_T = pts2[1, i] * pt_T

        # Add [pt_T, 0, -ui_pt_T]
        L[2 * i] = np.hstack((pt_T, np.zeros(3), -ui_pt_T))

        # Add [0, pt_T, -vi_pt_T]
        L[2 * i + 1] = np.hstack((np.zeros(3), pt_T, -vi_pt_T))

    U, S, Vh = np.linalg.svd(L)
    H = Vh[-1, :] / Vh[-1, -1]
    H = H.reshape(3, 3)
    return H


if __name__ == '__main__':
    pts1 = np.asarray([[1, 0], [2, 1], [3, 2], [4, 3]]).T
    pts2 = np.asarray([[3, 0], [4, 1], [5, 2], [6, 3]]).T
    print(pts1)
    print(pts2)
    H = computeH(pts1, pts2)
    print(H)

    pts1 = np.vstack((pts1, np.ones(pts1.shape[1])))
    x = np.matmul(H, pts1)
    xs = x[0, :] / x[2, :]
    ys = x[1, :] / x[2, :]
    print(x)
    print(xs)
    print(ys)
