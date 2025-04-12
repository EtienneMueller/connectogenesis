import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_coordinate(file_path, layer_shift = 15):
    length = len(np.load(file_path + f"/plane{0+layer_shift}/stat.npy", allow_pickle=True))
    x_array = np.empty(length)
    y_array = np.empty(length)
    z_array = np.empty(length)
    brightness = np.random.rand(length, 1212)
    for i in range(1):
        stat_file = file_path + f"/plane{i+layer_shift}/stat.npy"
        stat = np.load(stat_file, allow_pickle=True)
        print(len(stat))

        f_file = file_path + f"/plane{i+layer_shift}/F.npy"
        f = np.load(f_file, allow_pickle=True)
        
        for j in range(length):
            x_array[j] = stat[j]['med'][0]
            y_array[j] = stat[j]['med'][1]
            z_array[j] = i
            brightness[j] = f[j]

    b_min = np.min(brightness)
    b_max = np.max(brightness)
    brightness = np.divide(brightness-b_min, b_max-b_min)

    x, y, z, brightness = filter(x_array, y_array, z_array, brightness)

    return x, y, z, brightness


def filter(x, y, z, brightness, brightness_threshold=0.32):
    average_brightness = np.mean(brightness, axis=1)
    mask = average_brightness >= brightness_threshold
    x, y, z = x[mask], y[mask], z[mask]
    brightness = brightness[mask]

    b_min = np.min(brightness)
    b_max = np.max(brightness)
    brightness = np.divide(brightness-b_min, b_max-b_min)

    return x, y, z, brightness


def anim(x, y, z, brightness):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, s=brightness[:, 0]*100, c='blue', cmap='viridis')

    def update(frame):
        sizes = brightness[:, frame] * 250
        sc._sizes = sizes
        return sc,

    ani = animation.FuncAnimation(fig, update, frames=1212, interval=100, blit=True)
    plt.show()


def s2p_to_snn(x, y, z):
    pos = nest.spatial.free(pos=[[x, y, z] for x, y, z in zip(x, y, z)])
    s_nodes = nest.Create('iaf_psc_alpha', positions=pos)
    nest.PlotLayer(s_nodes)
    plt.show()


def main():
    x, y, z, brightness = load_coordinate('big_data/sample1')
    anim(x, y, z, brightness)
    #s2p_to_snn(x, y, z)


if __name__=='__main__':
    #nest.ResetKernel()
    main()
