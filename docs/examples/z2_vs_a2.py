# this file shows the comparison of the Z2 Lattice to the A2 Lattice

import latbin
import numpy as np

# create some fake data with shape (npts,ndim)
npts,ndim = 60000,2
data = np.random.normal(size=(npts, ndim))*4.0

# create a Z2 lattice (square lattice)
z2 = latbin.ZLattice(2)

# create an A2 lattice (extension to the honeycomb lattice)
a2 = latbin.ALattice(2)

# histogram the data onto both lattices
h1 = z2.histogram(data)
h2 = a2.histogram(data)

# get the lattice points in the data space
centers1 = h1.centers()
centers2 = h2.centers()

# show the result
import matplotlib.pylab as plt
fig,ax = plt.subplots(1,2,sharex=True,sharey=True)

ax[0].set_title("Square binning (Z2 Lattice)")
ax[0].scatter(centers1[:,0],centers1[:,1],c=h1.values(), s=70)

ax[1].set_title("Honeycomb binning (A2 Lattice)")
ax[1].scatter(centers2[:,0],centers2[:,1],c=h2.values(), s=70)

plt.show()

