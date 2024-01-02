import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    Ki = np.linalg.inv(K)
    a = np.linalg.norm(Pw[2, :] - Pw[3, :])
    b = np.linalg.norm(Pw[1, :] - Pw[3, :])
    c = np.linalg.norm(Pw[1, :] - Pw[2, :])
    # a = np.round(a, 1)
    # b = np.round(b, 1)
    # c = np.round(c, 1)
    # print("a", a)
    # print("b", b)
    # print("c", c)
    ph = np.array([[Pc[0, 0], Pc[0, 1], 1],
                   [Pc[1, 0], Pc[1, 1], 1],
                   [Pc[2, 0], Pc[2, 1], 1],
                   [Pc[3, 0], Pc[3, 1], 1]])
    pht = ph.T
    p = (Ki @ pht).T
    # n_1 = np.linalg.norm(p[1, :])
    # n_2 = np.linalg.norm(p[2, :])
    # n_3 = np.linalg.norm(p[3, :])
    # pu1 = p[1, :] / n_1
    # pu2 = p[2, :] / n_1
    # pu3 = p[3, :] / n_1
    pu1 = (1 / np.linalg.norm(p[1, :])) * p[1, :]
    pu2 = (1 / np.linalg.norm(p[2, :])) * p[2, :]
    pu3 = (1 / np.linalg.norm(p[3, :])) * p[3, :]
    # pu1 = np.round(pu1, 1)
    # pu2 = np.round(pu2, 1)
    # pu3 = np.round(pu3, 1)
    # print("pu1", pu1)
    # print("pu2", pu2)
    # print("pu3", pu3)
    cos_al = np.dot(pu2, pu3)
    cos_bt = np.dot(pu1, pu3)
    cos_gm = np.dot(pu1, pu2)
    # cos_al = np.round(cos_al, 4)
    # cos_bt = np.round(cos_bt, 4)
    # cos_gm = np.round(cos_gm, 4)
    # print("cos_al", cos_al)
    # print("cos_bt", cos_bt)
    # print("cos_gm", cos_gm)

    A0 = (1 + (a ** 2 - c ** 2) / b ** 2) ** 2 - 4 * (a ** 2 / b ** 2) * cos_gm ** 2
    A1 = 4 * (-((a ** 2 - c ** 2) / b ** 2) * (1 + ((a ** 2 - c ** 2) / b ** 2)) * cos_bt + 2 * (a ** 2 / b ** 2) *
              (cos_gm ** 2) * cos_bt - (1 - ((a ** 2 + c ** 2) / b ** 2)) * cos_al * cos_gm)
    A2 = 2 * (((a ** 2 - c ** 2) / b ** 2) ** 2 - 1 + 2 * ((a ** 2 - c ** 2) / b ** 2) ** 2 * cos_bt ** 2 + 2 *
              ((b ** 2 - c ** 2) / b ** 2) * cos_al ** 2 - 4 * (
                          (a ** 2 + c ** 2) / b ** 2) * cos_al * cos_bt * cos_gm + 2 *
              ((b ** 2 - a ** 2) / b ** 2) * cos_gm ** 2)
    A3 = 4 * (((a ** 2 - c ** 2) / b ** 2) * (1 - ((a ** 2 - c ** 2) / b ** 2)) * cos_bt -
              (1 - ((a ** 2 + c ** 2) / b ** 2)) * cos_al * cos_gm + 2 * (c ** 2 / b ** 2) * (cos_al ** 2) * cos_bt)
    A4 = ((a ** 2 - c ** 2) / b ** 2 - 1) ** 2 - 4 * (c ** 2 / b ** 2) * cos_al ** 2

    A = [A4, A3, A2, A1, A0]
    # A = np.round(A, 4)
    v = np.roots(A)
    v = np.array(v)
    v = v.real
    v = v[v > 0]
    # v = np.round(v, 5)
    # print("v", v)
    R_t = None
    t_t = None
    tolerance = 100000000000000000
    for i in range(len(v)):
        u = ((-1 + ((a ** 2 - c ** 2) / b ** 2)) * v[i] * v[i] - 2 * ((a ** 2 - c ** 2) / b ** 2) * cos_bt *
             v[i] + 1 + ((a ** 2 - c ** 2) / b ** 2)) / (2 * (cos_gm - v[i] * cos_al))
        # print("u", u)
        s1 = np.sqrt((b ** 2) / (1 + v[i] ** 2 - 2 * v[i] * cos_bt))
        s2 = u * s1
        s3 = v[i] * s1
        # print("s1", s1)
        # print("s2", s2)
        # print("s3", s3)
        P1 = s1 * pu1
        P2 = s2 * pu2
        P3 = s3 * pu3
        Pcam = np.stack((P1, P2, P3), axis=0)
        # print("Pcam", Pcam)

        R_cw, t_cw = Procrustes(Pw[1:], Pcam)
        # R_cw = np.linalg.inv(R_wc)
        # t_cw = -(R_cw @ t_wc)
        # print("R", R)
        # print("t", t)
        p_ref = Pc[0, :]
        Rwt = R_cw @ (Pw[0, :].T) + t_cw
        p_test = K @ Rwt
        p_test = p_test / p_test[2]
        p_test = np.delete(p_test, 2, 0)
        # print("p_test", p_test)
        error = np.linalg.norm(p_test - p_ref)
        # print("Error", error)
        if error < tolerance:
            tolerance = error
            R_t = R_cw
            t_t = t_cw
    R_ti = np.linalg.inv(R_t)

    R = R_ti
    # print("R_ti=", R_ti)
    t = -(R_ti @ t_t)
    # print("t", t)
    return R, t


def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    #Find centroid of X and Y
    Y_av = np.mean(Y, axis=0)
    X_av = np.mean(X, axis=0)
    A = (Y - Y_av).T
    B = (X - X_av).T
    #Find SVD:
    U, S, Vt = np.linalg.svd(A @ B.T, full_matrices=True)
    #Find R
        #Make sure det(R) = 1
    s = np.eye((3))
    s[2, 2] = np.linalg.det(Vt.T @ U.T)
    r = U @ s
    R_cw = r @ Vt
    #Find t from its definition
    t_cw = Y_av - R_cw @ X_av
    t_cw = np.reshape(t_cw, [3,])
    # R_wc = np.linalg.inv(R_cw)
    # t_wc = -(R_wc @ t_cw)
    return R_cw, t_cw