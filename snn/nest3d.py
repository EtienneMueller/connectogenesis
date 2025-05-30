import matplotlib.pyplot as plt
import nest

nest.ResetKernel()

pos = nest.spatial.free(nest.random.uniform(-0.5, 0.5), extent=[1.5, 1.5, 1.5])

l1 = nest.Create("iaf_psc_alpha", 1000, positions=pos)

# visualize

# extract position information, transpose to list of x, y and z positions
xpos, ypos, zpos = zip(*nest.GetPosition(l1))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xpos, ypos, zpos, s=15, facecolor="b")

# Gaussian connections in full box volume [-0.75,0.75]**3
nest.Connect(
    l1,
    l1,
    {
        "rule": "pairwise_bernoulli",
        "p": nest.spatial_distributions.gaussian(nest.spatial.distance, std=0.25),
        "allow_autapses": False,
        "mask": {"box": {"lower_left": [-0.75, -0.75, -0.75], "upper_right": [0.75, 0.75, 0.75]}},
    },
)

# show connections from center element
# sender shown in red, targets in green
ctr = nest.FindCenterElement(l1)
xtgt, ytgt, ztgt = zip(*nest.GetTargetPositions(ctr, l1)[0])
xctr, yctr, zctr = nest.GetPosition(ctr)
ax.scatter([xctr], [yctr], [zctr], s=40, facecolor="r")
ax.scatter(xtgt, ytgt, ztgt, s=40, facecolor="g", edgecolor="g")

tgts = nest.GetTargetNodes(ctr, l1)[0]
distances = nest.Distance(ctr, l1)
tgt_distances = [d for i, d in enumerate(distances) if i + 1 in tgts]

plt.figure()
plt.hist(tgt_distances, 25)
plt.show()