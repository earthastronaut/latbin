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
import pandas as pd
import matplotlib.pylab as plt

# create some fake data with shape (npts,ndim)
npts,ndim = 60000,2
scale = 4.0
data = pd.DataFrame(
    {
        "x1":scale*np.random.normal(size=(npts,)),
        "x2":scale*np.random.normal(size=(npts,)),
    }
)

# create an A2 lattice (honeycomb binning)
a2 = latbin.ALattice(2)

# bin the data onto the lattice
# the binned data is simply a pandas groupby object
binned_data = a2.bin(data)

#find the mean of the data in each bin
centers = binned_data.mean()

#find the number of data points in each bin
counts = binned_data.size()

# show the result
plt.title("Honeycomb binning (A2 Lattice)")
plt.scatter(centers["x1"],centers["x2"], c=counts, s=70)
plt.show()
```


## Installation


In the terminal you can install this in the usual way.

```bash
python setup.py install
```
