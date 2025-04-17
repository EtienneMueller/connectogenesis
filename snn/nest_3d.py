import matplotlib.pyplot as plt
import nest
#import mat73
import numpy as np


#mat = mat73.loadmat('data/Fish_03_ROI_centroids.mat')
#np.save('data/Fish03', mat['ROI_centroids'])

mat = np.load('data/Fish03.npy')
mat = mat.tolist()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# x = mat[:, 0]
# y = mat[:, 1]
# z = mat[:, 2]

# ax.scatter(x, y, z, marker=',')
# plt.show()

nest.ResetKernel()

#pos = nest.spatial.free(nest.random.uniform(-0.5, 0.5), extent=[1.5, 1.5, 1.5])
#pos = nest.spatial.free(pos=[[1,2,3], [1,5,6], [1,2,9], [10,11,12], [13,14,15]])
pos = nest.spatial.free(pos=mat)

l1 = nest.Create("iaf_psc_alpha", positions=pos)

# visualize

# extract position information, transpose to list of x, y and z positions
xpos, ypos, zpos = zip(*nest.GetPosition(l1))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xpos, ypos, zpos, s=1, facecolor="b")

# full connections in box volume [-0.2,0.2]**3
nest.Connect(
    l1,
    l1,
    {
        "rule": "pairwise_bernoulli",
        "p": 1.0,
        "allow_autapses": False,
        "mask": {"box": {"lower_left": [-0.2, -0.2, -0.2], "upper_right": [0.2, 0.2, 0.2]}},
    },
)

# show connections from center element
# sender shown in red, targets in green
ctr = nest.FindCenterElement(l1)
#xtgt, ytgt, ztgt = zip(*nest.GetTargetPositions(ctr, l1)[0])
xctr, yctr, zctr = nest.GetPosition(ctr)
ax.scatter([xctr], [yctr], [zctr], s=40, facecolor="r")
#ax.scatter(xtgt, ytgt, ztgt, s=40, facecolor="g", edgecolor="g")

tgts = nest.GetTargetNodes(ctr, l1)[0]
distances = nest.Distance(ctr, l1)
tgt_distances = [d for i, d in enumerate(distances) if i + 1 in tgts]

plt.figure()
plt.hist(tgt_distances, 25)
plt.show()
