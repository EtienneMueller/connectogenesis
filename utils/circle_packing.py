import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from descartes import PolygonPatch


def create_random_polygon(num_vertices=10, radius=10):
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    points = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    np.random.shuffle(points)
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if isinstance(polygon, MultiPolygon):
        # Choose the largest polygon if it's a MultiPolygon
        polygon = max(polygon, key=lambda p: p.area)
    return orient(polygon)


def pack_circles(polygon, radius=1, spacing=0.1):
    minx, miny, maxx, maxy = polygon.bounds
    circles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            point = Point(x, y)
            if polygon.contains(point.buffer(radius)):
                circles.append(point)
            y += 2 * radius + spacing
        x += 2 * radius + spacing
    return circles


def plot_packing(polygon, circles, radius):
    fig, ax = plt.subplots()
    patch = PolygonPatch(polygon, facecolor='lightblue', edgecolor='black', alpha=0.5)
    ax.add_patch(patch)
    for circle in circles:
        circle_patch = plt.Circle((circle.x, circle.y), radius, edgecolor='black', facecolor='none')
        ax.add_patch(circle_patch)
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()


def main():
    random_polygon = create_random_polygon()
    circle_radius = 0.5
    circles = pack_circles(random_polygon, radius=circle_radius)
    plot_packing(random_polygon, circles, radius=circle_radius)


if __name__ == "__main__":
    main()
