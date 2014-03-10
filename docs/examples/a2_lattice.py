import latbin
import numpy as np

# create some fake data with shape (npts,ndim)
npts,ndim = 60000,2
data = np.random.normal(size=(npts, ndim))*4.0

# create an A2 lattice (honeycomb binning)
a2 = latbin.ALattice(2)

# histogram the data onto A2 Lattice
h = a2.histogram(data)

# get the lattice points in the data space
centers = h.centers()

# show the result
import matplotlib.pylab as plt
plt.title("Honeycomb binning (A2 Lattice)")
plt.scatter(centers[:,0],centers[:,1],c=h.values(), s=70)
plt.show()

