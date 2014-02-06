# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Standard Library
import unittest

# 3rd Party

# Internal
import latbin

# ########################################################################### #

class TestCore (unittest.TestCase):
    
    def __setUp__ (self):
        super(TestCore, self).__setUp__()
        self.random_2d_pts = 20.0*(np.random.random((100, 1))-0.5)
        self.random_2d_pts = 20.0*(np.random.random((100, 2))-0.5)
        self.random_3d_pts = 20.0*(np.random.random((100, 3))-0.5)
        self.random_4d_pts = 20.0*(np.random.random((100, 4))-0.5)
    
    def test_quantize(self):
        


# ########################################################################### #

if __name__ == "__main__":
    unittest.main()
