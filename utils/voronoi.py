import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def main():
    points = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 1.5],
        [4.0, 4.0],
        [5.0, 3.5],
        [6.0, 2.0],
    ])

    vor = Voronoi(points)

    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False)

    circle_radius = 0.3
    for point in points:
        circle = plt.Circle(point, circle_radius, fill=False, edgecolor='b')
        ax.add_artist(circle)

    # HERE
    max_radius = 0
    best_center = None

    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            polygon = np.array(polygon)
            # new center and radius
            distances = [np.linalg.norm(polygon - point, axis=1) for point in points]
            min_distances = np.min(distances, axis=0)
            center = vor.points[np.argmin(min_distances)]
            radius = min(min_distances)
            if radius > max_radius:
                max_radius = radius
                best_center = center

    if best_center is not None:
        new_circle = plt.Circle(best_center, max_radius, color='r', fill=False, linestyle='--')
        ax.add_artist(new_circle)
        print(f"Largest empty circle center: {best_center}, radius: {max_radius}")

    plt.xlim(0, 7)
    plt.ylim(0, 5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    main()
