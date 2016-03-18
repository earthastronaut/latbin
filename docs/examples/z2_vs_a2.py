# this file shows the comparison of the Z2 Lattice to the A2 Lattice

import latbin
import numpy as np
import matplotlib.pyplot as plt

# create some fake data with shape (npts,ndim)
npts,ndim = 60000,2
data = np.random.normal(size=(npts, ndim))*4.0

# create a Z2 lattice (square lattice)
z2 = latbin.ZLattice(2)

# create an A2 lattice (extension to the honeycomb lattice)
a2 = latbin.ALattice(2)

# bin the data onto the lattices
h1 = z2.bin(data)
h2 = a2.bin(data)

# get the mean of the data in each bin
centers1 = h1[[0, 1]].mean().values
centers2 = h2[[0, 1]].mean().values
#find the number of points in each bin
n1 = h1.size().values
n2 = h2.size().values

# show the result
fig,ax = plt.subplots(1,2,sharex=True,sharey=True)

ax[0].set_title("Square binning (Z2 Lattice)")
ax[0].scatter(centers1[:,0],centers1[:,1],c=n1, s=70)

ax[1].set_title("Honeycomb binning (A2 Lattice)")
ax[1].scatter(centers2[:,0],centers2[:,1],c=n2, s=70)

plt.show()

