import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


np.random.seed(42)
num_points_a = 80
num_points_b = 100


def generate_data():
    points_a = np.random.rand(num_points_a, 2)
    points_b = np.random.rand(num_points_b, 2)
    return points_a, points_b


def print_result(points_a, points_b, row_indices, col_indices, dist_matrix, valid_matches):
    #for i, (row, col) in enumerate(zip(row_indices, col_indices)):
    #    print(f"A: {points_a[row]}   B: {points_b[col]}   Distance {dist_matrix[row][col]}")
    # Print the results
    for row, col in zip(row_indices[valid_matches], col_indices[valid_matches]):
        print(f"A: {points_a[row]}   B: {points_b[col]}   Distance {dist_matrix[row][col]}")


def plot_result(points_a, points_b, row_indices, col_indices, valid_matches):
    # plot points and corresponding lines
    plt.figure(figsize=(8, 6))
    plt.scatter(points_a[:, 0], points_a[:, 1], c='blue', label='Graph A')
    plt.scatter(points_b[:, 0], points_b[:, 1], c='red', label='Graph B')

    # draw lines between corresponding points
    for row, col in zip(row_indices[valid_matches], col_indices[valid_matches]):
        plt.plot([points_a[row][0], points_b[col][0]], [points_a[row][1], points_b[col][1]], 'k--')

    plt.legend()
    plt.title('Uneven matching')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def main():
    points_a, points_b = generate_data()
    dist_matrix = distance_matrix(points_a, points_b)
    # add a dummy row with large distances
    large_distance = 1e6
    dummy_row = np.full((1, num_points_b), large_distance)
    dist_matrix_with_dummy = np.vstack([dist_matrix, dummy_row])

    # hungarian
    row_indices, col_indices = linear_sum_assignment(dist_matrix_with_dummy)

    # filter out dummys
    valid_matches = row_indices < num_points_a

    print_result(points_a, points_b, row_indices, col_indices, dist_matrix, valid_matches)
    plot_result(points_a, points_b, row_indices, col_indices, valid_matches)


if __name__ == "__main__":
    main()
