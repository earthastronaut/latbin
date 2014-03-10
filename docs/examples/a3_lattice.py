import latbin
import numpy as np

# create some fake data
npts,ndim = 5500,3
data = np.random.normal(size=(npts, ndim))*2.0

# create an A3 lattice (extension to the honeycomb lattice)
a3 = latbin.ALattice(ndim)

# histogram the data
hist = a3.histogram(data)
centers = hist.centers()

# show the result
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
ax = plt.figure().gca(projection='3d')
ax.scatter(centers[:,0],centers[:,1],centers[:,2],c=hist.values())
plt.show()
