***********************
Data Comparison Example
***********************

Often when you bin two data sets you want to be able to compare them. This one of the primary goals with latbin and reasons for development because no other code existed for this. The example :download:`data_comparisons.py <data_comparisons.py>` shows a simple way to compare data. Walking through the code there's first the import statements and data creation:

.. code:: python

   import latbin
   import numpy as np

   # create fake data sets
   dat1 = np.random.uniform(-10,10,size=(5000, 2))
   dat2 = np.random.normal(size=(5700, 2))*8.0

Then we're going to create some lattice object to histogram our data onto. In this example, because it's easy to plot latter, we're going to use the :math:`A^2` lattice. The :math:`A^2` lattice should be familiar as a hexagonal or honeycomb lattice. Here we create the lattice and histogram our data sets:

.. code:: python

   # create an A2 Lattice (honeycomb lattice)
   a2 = latbin.ALattice(2)

   # histogram the data sets
   h1 = a2.histogram(dat1,norm=True)
   h2 = a2.histogram(dat2,norm=True)

What the method `labin.Lattice.histogram` returns is a `latbin.PointInformation` object. At it's core this object is a dictionary who's keys are the unique lattice points which data is quantized onto. The values of the `latbin.PointInformation` is the histogrammed value, in this case the point density. Here we do subtraction but other operators are supported (e.g. *,+,/) and there's even a method `latbin.PointInformation.operation` which will take an arbitrary function and apply it to the two data sets when the keys match. 

.. code:: python

   # do some operation to the data sets
   diff = h1-h2

The variable `diff` is also a `latbin.PointInformation` object. We can get the lattice centers in the data space and plot the values.

.. code:: python

   # get the data centers
   centers = diff.centers()

   # show the results
   import matplotlib.pylab as plt
   plt.colorbar(plt.scatter(centers[:,0], centers[:,1], c=diff.values()))
   plt.show()


:download:`(source) <data_comparisons.py>`

.. another method to show the code
.. .. literalinclude:: a2_lattice.py
..    :linenos:
..    :language: python
.. 



