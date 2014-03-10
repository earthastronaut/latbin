import latbin
import numpy as np

# create fake data sets
dat1 = np.random.uniform(-10,10,size=(5000, 2))
dat2 = np.random.normal(size=(5700, 2))*8.0

# create an A2 Lattice (honeycomb lattice)
a2 = latbin.ALattice(2)

# histogram the data sets
h1 = a2.histogram(dat1,norm=True)
h2 = a2.histogram(dat2,norm=True)

# do some operation to the data sets
diff = h1-h2

# get the data centers
centers = diff.centers()
x_centers = centers[:, 0]
y_centers = centers[:, 1]

# show the results
import matplotlib.pylab as plt
plt.colorbar(plt.scatter(x_centers, y_centers, c=diff.values()))
plt.show()
