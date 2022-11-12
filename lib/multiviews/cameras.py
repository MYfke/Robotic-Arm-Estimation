import numpy as np


def unfold_camera_param(camera):
    """
    解析相机的参数
    """

    R = camera['R']
    T = camera['T']
    f = 0.5 * (camera['fx'] + camera['fy'])
    c = np.array([camera['cx'], camera['cy']])
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def project_point_radial(x, R, T, f, c, k, p):
    """
    将世界坐标转化为像素坐标

    Args:
        x: 世界坐标中的 N 个三维坐标点
        R: 3x3 相机旋转矩阵
        f: (标量)相机焦距
        c: 2x1 相机中心
        k: 3x1 相机径向畸变系数
        p: 2x1 相机切向畸变系数
    Returns:
        ypixel.T: Nx2 像素空间中的 N 个点
    """

    n = x.shape[0]
    xcam = R.dot(x.T - T)
    y = xcam[:2] / xcam[2]

    r2 = np.sum(y**2, axis=0)
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                           np.array([r2, r2**2, r2**3]))
    tan = 2 * p[0] * y[1] + 2 * p[1] * y[0]
    y = y * np.tile(radial + tan,
                    (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    ypixel = (f * y) + c
    return ypixel.T


def project_pose(x, camera):
    """
    传入三维点的列表和一个相机实例，输出像素坐标列表
    """

    R, T, f, c, k, p = unfold_camera_param(camera)
    return project_point_radial(x, R, T, f, c, k, p)


def world_to_camera_frame(x, R, T):
    """
    世界坐标转相机坐标

    Args:
        x: Nx3 3d points in world coordinates  世界坐标
        R: 3x3 Camera rotation matrix  相机坐标到世界坐标的旋转矩阵
        T: 3x1 Camera translation parameters  相机坐标到世界坐标的平移矩阵
    Returns:
        xcam: Nx3 3d points in camera coordinates  相机坐标
    """

    xcam = R.dot(x.T - T)  # rotate and translate
    return xcam.T


def camera_to_world_frame(x, R, T):
    """
    相机坐标转世界坐标

    Args:
        x: Nx3 points in camera coordinates  相机坐标
        R: 3x3 Camera rotation matrix  相机坐标到世界坐标的旋转矩阵
        T: 3x1 Camera translation parameters  相机坐标到世界坐标的平移矩阵
    Returns:
        xwrd: Nx3 points in world coordinates  世界坐标
    """

    xwrd = R.T.dot(x.T) + T  # rotate and translate
    return xwrd.T
