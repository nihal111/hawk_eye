import numpy as np


def slope_and_intercept(point1, point2):
    m = (point1[1] - point2[1]) * 1.0 / (point1[0] - point2[0])
    c = point1[1] - m * point1[0]
    return m, c


def find_intersection(m1, c1, m2, c2):
    x = (c2 - c1) * 1.0 / (m1 - m2)
    y = m1 * x + c1
    return np.array([x, y])


def find_polar(point, origin):
    diff = point - origin
    r = np.sqrt(np.sum(np.square(diff)))
    theta = np.arctan2(diff[1], diff[0])
    return r, theta


def find_cartesian(r, theta, origin):
    point = np.array([r * np.cos(theta), r * np.sin(theta)])
    return point + origin


def transform(r, theta, delta_theta):
    return r, theta + delta_theta


def pan(points, delta_theta=0.087):
    # Take Note: points are not homogenous, np.array([x, y]) form
    x0, y0 = points[0]  # p0 - TL
    x1, y1 = points[1]  # p1 - TR
    x2, y2 = points[2]  # p2 - BR
    x3, y3 = points[3]  # p3 - BL

    # Find L1
    m1, c1 = slope_and_intercept(points[0], points[3])

    # Find L2
    m2, c2 = slope_and_intercept(points[1], points[2])

    C = find_intersection(m1, c1, m2, c2)

    new_points = []
    for point in points:
        r, theta = find_polar(point, C)
        r, theta = transform(r, theta, delta_theta)
        x, y = find_cartesian(r, theta, C)
        new_points.append([x, y])

    return new_points


if __name__ == '__main__':
    print(pan([np.array([5, 8.66]), np.array([8.66, 5]),
               np.array([4.33, 2.5]), np.array([2.5, 4.33])], 0.087))
