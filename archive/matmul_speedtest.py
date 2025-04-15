import numpy as np
import time


def main():
    size = 30000
    a = np.random.randn(size)
    b = np.random.randn(size)
    m = np.random.randn(size, size)

    # 10,000 = 781 MB
    # 20,000 (400M) = 3.00 GB (7.5 kB/1k) (9.46 GB)
    # 25,000 (625M) = 4.68 GB (7.5)
    # 30,000 (900M) = 6.73 (25-26 GB)
    # 35,000 = 9.15 25.13 GB
    # 40,000 = ~35 GB

    print("rand done")
    time.sleep(15)

    print("start")
    start_time = time.time()

    for i in range(10):
        c = np.multiply(a, m)
        d = np.add(c, b)
        # j = 100
        #if i%j==0:
        #    print(i/j)

    calctime = (time.time() - start_time)
    print(calctime, "seconds")
    print(c.shape, d.shape)


if __name__ == "__main__":
    main()
