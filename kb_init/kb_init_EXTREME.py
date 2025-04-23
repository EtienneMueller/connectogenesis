import numpy as np
import matplotlib.pyplot as plt
import slits as load_s2p


def kb_init(x, y, activity, min_threshold=0.0, max_threshold=0.3, max_distance=10):
    num_time_points = activity.shape[1]
    num_points = len(x)
    
    X, Y = np.meshgrid(x, x)
    distances = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
    
    lines = []
    alphas = []
    
    for t in range(num_time_points):
        if t%20==0:
            print(f"Step {t} of {activity.shape[1]}")
        for i in range(num_points):
            for j in range(i + 1, num_points):
                if distances[i, j] < max_distance:
                    avg_activity = (activity[i, t] + activity[j, t]) / 2
                    if avg_activity > min_threshold:
                        # thickness = (avg_activity - min_threshold) / (max_threshold - min_threshold)
                        # thickness = max(0, min(thickness, 1))  # Clamping thickness to [0, 1]
                        # lines.append(([x[i], x[j]], [y[i], y[j]]))
                        # plt.plot([x[i], x[j]], [y[i], y[j]], 'r', alpha=0.5, linewidth=thickness * 5)  # Scale thickness
                        alpha = (avg_activity - min_threshold) / (max_threshold - min_threshold)
                        alpha = max(0, min(alpha, 1))  # clamping alpha to [0, 1]
                        lines.append(([x[i], x[j]], [y[i], y[j]]))
                        alphas.append(alpha)
    return lines, alphas
    

def main():
    x, y, _, activity = load_s2p.load_coordinate('big_data/sample1')
    plt.figure()
    plt.plot(x, y, 'o')
    lines, alphas = kb_init(x, y, activity)
    print("plotting...")
    for line, alpha in zip(lines, alphas):
        plt.plot(line[0], line[1], 'r', alpha=alpha)  # transparency
    plt.show()


if __name__=='__main__':
    main()
