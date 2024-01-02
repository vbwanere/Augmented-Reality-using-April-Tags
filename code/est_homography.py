import numpy as np


def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneous transformation matrix s.t. Y ~ H*X

    """

    X1, X2, X3, X4 = X[0, 0], X[1, 0], X[2, 0], X[3, 0]
    Y1, Y2, Y3, Y4 = X[0, 1], X[1, 1], X[2, 1], X[3, 1]
    Xd1, Xd2, Xd3, Xd4 = Y[0, 0], Y[1, 0], Y[2, 0], Y[3, 0]
    Yd1, Yd2, Yd3, Yd4 = Y[0, 1], Y[1, 1], Y[2, 1], Y[3, 1]

    A = np.array([[-X1, -Y1, -1, 0, 0, 0, X1 * Xd1, Y1 * Xd1, Xd1],
                  [0, 0, 0, -X1, -Y1, -1, X1 * Yd1, Y1 * Yd1, Yd1],
                  [-X2, -Y2, -1, 0, 0, 0, X2 * Xd2, Y2 * Xd2, Xd2],
                  [0, 0, 0, -X2, -Y2, -1, X2 * Yd2, Y2 * Yd2, Yd2],
                  [-X3, -Y3, -1, 0, 0, 0, X3 * Xd3, Y3 * Xd3, Xd3],
                  [0, 0, 0, -X3, -Y3, -1, X3 * Yd3, Y3 * Yd3, Yd3],
                  [-X4, -Y4, -1, 0, 0, 0, X4 * Xd4, Y4 * Xd4, Xd4],
                  [0, 0, 0, -X4, -Y4, -1, X4 * Yd4, Y4 * Yd4, Yd4]])

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    h = []
    for i in range(0, 9):
        a = vh[8, i]
        h.append(a)
    h = np.array(h)
    H = np.reshape(h, (3, 3))

    return H
