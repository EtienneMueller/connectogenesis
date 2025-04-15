import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


# https://en.wikipedia.org/wiki/Hungarian_algorithm
np.random.seed(42)
num_points = 100


def generate_data():
    points_a = np.random.rand(num_points, 2)
    points_b = np.random.rand(num_points, 2)
    return points_a, points_b


def print_resutl(points_a, points_b, row_indices, col_indices, dist_matrix):
    for i, (row, col) in enumerate(zip(row_indices, col_indices)):
        print(f"A: {points_a[row]}   B: {points_b[col]}   Distance {dist_matrix[row][col]}")


def plot_result(points_a, points_b, row_indices, col_indices):
    plt.figure(figsize=(8, 6))
    plt.scatter(points_a[:, 0], points_a[:, 1], c='blue', label='Graph A')
    plt.scatter(points_b[:, 0], points_b[:, 1], c='red', label='Graph B')

    # lines between corresponding points
    for i, (row, col) in enumerate(zip(row_indices, col_indices)):
        plt.plot([points_a[row][0], points_b[col][0]], [points_a[row][1], points_b[col][1]], 'k--')

    plt.legend()
    plt.title('Even matching')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def main():
    points_a, points_b = generate_data()
    dist_matrix = distance_matrix(points_a, points_b)
    
    # hungarian
    row_indices, col_indices = linear_sum_assignment(dist_matrix)

    print_resutl(points_a, points_b, row_indices, col_indices, dist_matrix)
    plot_result(points_a, points_b, row_indices, col_indices)


if __name__ == "__main__":
    main()
