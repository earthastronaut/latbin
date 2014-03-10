# LATBIN #

This python package is used for binning onto lattices in multiple dimensions. 

## Installation ##

In the terminal you can install this in the usual way.

    python setup.py install

## Basic Example ##

This example shows two data sets which have various lengths but both have two dimensions. We create an A2 (aka honeycomb) lattice which we histogram both data sets onto. Then we can make comparisons (e.g. by subtraction) of the two and analyze the result. 

    import latbin
    import numpy as np

    # create fake data sets
    dat1 = np.random.uniform(-10,10,size=(2000, 2))
    dat2 = np.random.normal(size=(2300, 2))*8.0
    
    # create an A2 Lattice (honeycomb lattice)
    lat = latbin.ALattice(2)
    
    # histogram the data sets
    h1 = lat.histogram(dat1)
    h2 = lat.histogram(dat2)
    
    # do some operation to the data sets
    diff = h1-h2
    
    # get the data centers
    centers = diff.centers()
    x_centers = centers[:, 0]
    y_centers = centers[:, 1]
    
    # show the results
    import matplotlib.pylab as plt
    plt.scatter(x_centers, y_centers, c=diff.values())
    plt.show()

