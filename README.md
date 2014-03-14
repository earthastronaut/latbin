# LATBIN #

This python package is used for binning onto lattices in multiple dimensions. 


## Why Use Lattices 


For "regular" 2D histogramming you implicitly use a Z2 lattice! But the errors involved with quantizing to a Z2 lattices is larger than A2.

insert plot here showing proper comparison of Z2 and A2
give name of file which creates Z2-A2 comparison


## Basic Example


This example shows how to take a normal data set and histogram the data onto
an A2 (aka honeycomb) lattice using latbin. 

```python
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
```


## Installation


In the terminal you can install this in the usual way.

```bash
python setup.py install
```
