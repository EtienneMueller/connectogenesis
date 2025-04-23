import nest
import numpy as np
import matplotlib.pyplot as plt
import slits   


def main():
    x, y, z, _ = slits.load_coordinate('big_data/sample1')
    pos = nest.spatial.free(pos=[[x, y, z] for x, y, z in zip(x, y, z)])
    s_nodes = nest.Create('iaf_psc_alpha', positions=pos)
    nest.PlotLayer(s_nodes)
    plt.show()


if __name__=='__main__':
    nest.ResetKernel()
    main()
