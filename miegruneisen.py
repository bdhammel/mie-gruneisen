import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
#from scipy.constants import h, k
from scipy.misc import derivative


"""

References
--------
[1] "Equation of state modeling," in Selected Topics in Shock Wave Physics and 
Equation of State Modeling, pp. 155–227, WORLD SCIENTIFIC, apr 1994.
doi: 10.1142/9789814350273_0012

[2]
"""

# System parameters and constants 
# -----------------------
beta = .30 #  = 1/(k_b * T)  units of J
h = 1 # Plank's constant
V0 = 13 # normalized to Plank's constant i.e. V = V'/h
V = np.linspace(V0/2, V0, 100) # generate compressional states (volume) of the system
epsilon = 1-V/V0
gamma = 2 #  under the assumption that gamma remains constant
# -----------------------

# planks constant * simple harmonic oscillator frequency
# such that nu = exp[ - gamma * ln V ] based on equation [1 - 12.1.12 ]
hnu = lambda v: h*np.exp(-gamma*np.log(v)) 

# Energy state of a SHO
Ei = lambda i, v: (.5 + i)*hnu(v)

def internal_energy(V):
    """Average internal energy of the system

    E = E(V)  Internal energy states of the SHO change depending on volume

    Calculates: 
    <E> = sum [ E_i exp( - E_i * beta ) ]/ Z

    w/ Z = Z(V) = cononical enbsemble

    Args
    ---
    V (float) : Volume of the system, 

    """
    return float(mp.nsum(
        lambda x: Ei(x, V) * mp.exp(-beta*Ei(x, V)), [0,mp.inf], method='s+r') 
        / canonical_ensemble(V) )

def canonical_ensemble(V):
    """Calculate the cononical ensemble of the system

    Z = sum [ exp( - E_i * beta ) ]

    Args
    ----
    V (float) : Volume of the system, 
    """
    return float(mp.nsum(lambda x: mp.exp(-beta*Ei(x, V)), [0,mp.inf], method='s+r'))

def free_energy(V):
    """Calculate the Helmholtz free energy of the system 

    F = - 1/beta * ln (Z)

    Args
    ----
    V (float) : Volume of the system, 
    """
    return -1/beta*np.log(canonical_ensemble(V))

# Get the values for the canonical ensemble, internal energy, and free energy
# for each volume (compressional state) of the system  
Z = np.array([canonical_ensemble(Vi) for Vi in V])
E = np.array([internal_energy(Vi) for Vi in V])
F = np.array([free_energy(Vi) for Vi in V])

if __name__ == "__main__":

    dFdV = np.array([derivative(free_energy, Vi) for Vi in V])
    p = -dFdV

    hugoniot = 2*(E-E[0])/(V[0] - V) + p[0]

    d2FdV2 = np.array([derivative(free_energy, Vi, n=2) for Vi in V])
    K = V * d2FdV2
    hugoniot_analytic = K*epsilon / (1-(1+gamma/2)*epsilon)

    plt.ion()
    plt.close('all')
    plt.plot(V, hugoniot)
    plt.plot(V, hugoniot_analytic)


