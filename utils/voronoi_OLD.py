import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import combinations


def circle_intersections(c1, c2):
    (x1, y1, r1), (x2, y2, r2) = c1, c2
    d = np.hypot(x2 - x1, y2 - y1)
    if d > r1 + r2 or d < abs(r1 - r2) or (d == 0 and r1 == r2):
        return []  # No intersection

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - a**2)
    xm = x1 + a * (x2 - x1) / d
    ym = y1 + a * (y2 - y1) / d
    xs1 = xm + h * (y2 - y1) / d
    xs2 = xm - h * (y2 - y1) / d
    ys1 = ym - h * (x2 - x1) / d
    ys2 = ym + h * (x2 - x1) / d
    
    return [(xs1, ys1), (xs2, ys2)]


def find_tangent_points(circle1, circle2):
    (x1, y1, r1), (x2, y2, r2) = circle1, circle2
    d = np.hypot(x2 - x1, y2 - y1)
    if d > r1 + r2:
        raise ValueError("Circles are too far apart")
    
    # tangent points
    angle = np.arctan2(y2 - y1, x2 - x1)
    theta1 = np.arccos((r1 - r2) / d)
    theta2 = -theta1
    
    t1 = (x1 + r1 * np.cos(angle + theta1), y1 + r1 * np.sin(angle + theta1))
    t2 = (x2 + r2 * np.cos(angle + theta1), y2 + r2 * np.sin(angle + theta1))
    t3 = (x1 + r1 * np.cos(angle + theta2), y1 + r1 * np.sin(angle + theta2))
    t4 = (x2 + r2 * np.cos(angle + theta2), y2 + r2 * np.sin(angle + theta2))
    
    return [t1, t2, t3, t4]


def main():
    circles = [
        (1, 1, 1),  # (x, y, r)
        (4, 1, 1),
        (2.5, 3, 1)
    ]

    # intersection and tangent points
    points = []
    for (c1, c2) in combinations(circles, 2):
        points.extend(circle_intersections(c1, c2))
        if len(circle_intersections(c1, c2)) < 2:
            try:
                points.extend(find_tangent_points(c1, c2))
            except ValueError:
                continue

    if len(points) < 3:
        raise ValueError("Not enough points to form a polygon")

    points = np.array(points)
    hull = ConvexHull(points)

    plt.figure(figsize=(8, 8))
    for x, y, r in circles:
        circle = plt.Circle((x, y), r, edgecolor='b', facecolor='none')
        plt.gca().add_patch(circle)

    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
    plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')  # start point

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Convex Hull touching the inside of the circles')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()