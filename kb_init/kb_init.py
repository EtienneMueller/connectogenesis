import numpy as np
import matplotlib.pyplot as plt
import slits as load_s2p


threshold = 0.5
max_distance = 100  # 10


def kb_init(x, y, activity):
    plt.plot(x, y, 'o')
    for t in range(activity.shape[1]):
        if t%20==0:
            print(f"Step {t} of {activity.shape[1]}")
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if activity[i, t] > threshold and activity[j, t] > threshold:
                    distance = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                    if distance < max_distance:
                        plt.plot([x[i], x[j]], [y[i], y[j]], 'r', alpha=0.5)
    plt.show()


if __name__=='__main__':
    x, y, _, activity = load_s2p.load_coordinate('big_data/sample1')
    kb_init(x, y, activity)
