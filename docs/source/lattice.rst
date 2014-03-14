**************
Lattice Module
**************

.. automodule:: latbin.lattice

.. THE SO-CALLED root lattices are the n-dimensional lattices A, (n 2 l), o,(n 2 2), and E,(n = 6,7,8) defined in Section II. 

.. These lattices and their duals give rise
.. to the densest known sphere packing and coverings in dimensions n I	8, -and they can be used as the basis for efficient block quantizers for uniformly distributed inputs and to construct codes for a band-limited channel with Gaussiannoise(see[6], [9], [111,[161).Around eachlattice point is its Voronoi region, consisting of all points of the underlying spacewhich are closer to that lattice point than to any other. (Voronoi regions are also called Dirichlet regions, Brillouin zones, Wigner-Seitz cells, or nearest neighbor regions.) If the lattice is used as a quantizer, all the points in the Voronoi region around the lattice point x are representedby x; while if the lattice is used as a code for a Gaussianchannel,all the points in the Vordnoi region around x are decoded as x. In the preceding paper [6] we found the Voronoi regions for most of the root lattices and their duals, as well as the mean-squaredquantization error when theselattices are used to quantize uniformly distrib- uted data.

The abstract lattice class used in `latbin` is

.. autoclass:: latbin.lattice.Lattice


Lattice Classes
---------------

The following are specific classes of lattices which subclass the abstract lattice class. 


Z Lattice : *n*-dimensional integer lattice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: latbin.lattice.ZLattice
   :members: data_to_lattice_space, lattice_to_data_space, quantize, histogram, representation_to_centers

    
D Lattice : even sum integer lattice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: latbin.lattice.DLattice
   :members: data_to_lattice_space, lattice_to_data_space, quantize, histogram, representation_to_centers

   
A Lattice : integer coordinates that sum to zero
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: latbin.lattice.ALattice
   :members: data_to_lattice_space, lattice_to_data_space, quantize, histogram, representation_to_centers


Point Set Lattice : arbitrary lattice centers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: latbin.lattice.PointSet
   :members: quantize, histogram, count, index

.. include:: ../references.rst

