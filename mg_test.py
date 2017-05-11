import unittest
import numpy as np
from importlib import reload
try:
    miegruneisen
except:
    import miegruneisen 
else:
    reload(miegruneisen)

beta = miegruneisen.beta
hnu = miegruneisen.hnu
V = miegruneisen.V

SIG_FIG_ACCURACY = 10 

class MieGruneisenTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.Z = miegruneisen.Z
        cls.E = miegruneisen.E
        cls.F = miegruneisen.F

    def test_canonical_ensemble(self):
        """Check that the calculated value for the canonical ensemble matches 
        with the analytic expression

        Z = 2 sinh( h*nu / (s * k * T) )
        From equation 12.1.5
        """
        Z_analytic = (2*np.sinh(beta*hnu(V)/2))**(-1)
        np.testing.assert_array_almost_equal(self.Z, Z_analytic, 
                decimal=SIG_FIG_ACCURACY)

    def test_internal_energy(self):
        """Check the calculated internal energy values against the analytic values

        Check against eqns: 12.1.7 and 12.1.6
        """
        E_analytic = hnu(V)/2 * (np.tanh(beta*hnu(V)/2))**(-1)
        E_analytic2 = hnu(V)/2 + hnu(V) / (np.exp(beta*hnu(V))-1)
        np.testing.assert_almost_equal(E_analytic, E_analytic2, 
                decimal=SIG_FIG_ACCURACY)
        np.testing.assert_array_almost_equal(self.E, E_analytic, 
                decimal=SIG_FIG_ACCURACY)

    def test_free_energy(self):
        """Check the calculated free energy values against the analytic values

        Check against eqns: 12.1.7 and 12.1.8
        """
        F_analytic = hnu(V)/2 + 1/beta * np.log(1-np.exp(-beta*hnu(V)))
        np.testing.assert_array_almost_equal(self.F, F_analytic, 
                decimal=SIG_FIG_ACCURACY)

if __name__ == "__main__":
    unittest.main()
