#!/usr/bin/env python
# coding: utf-8

# Testing the renormalised Numerov method as described by Millard Alexander
# on a two channel scattering calculation of an Cl atom colliding with H2.


import matplotlib.pyplot as plt
import numpy as np

import qdyn
from qdyn.scattering import *


# constants
convE = 1./219474.6 # convert cminv to a.u.
convm = 1822.889    # convert atomic mass units to a.u.


# setting up the grid:
Nr = 200
r = np.linspace(3.5, 22, Nr)
h = r[2] - r[1]
r = np.append(r, r[-1]+h)
print('h = ', h)


# potential
def pot(r, l1, l2, l3, C1, C2, C3, C4):
    return C1*np.exp(-l1*r) + (C2+C3*r)*np.exp(-l2*r) - 0.5*C4*(np.tanh(1.2*(r-l3))+1)*r**(-6)

A = 293.3 * convE # spin-orbit constant

Sigma = pot(r, 0.813, 1.2014, 5.5701,
            3.7457e+3*convE, 6.7280e+5*convE, -1.2773e+5*convE, 3.4733e+6*convE)

Pi = pot(r, 0.677, 2.2061, 6.2093,
         -7.7718e+3*convE, -7.3837e+7*convE, 3.2149e+7*convE, 2.9743e+6*convE)

V11 = 2./3.*Sigma + Pi/3. - A
V12 = np.sqrt(2.)/3.*(Pi - Sigma)
V21 = V12
V22 = Sigma/3. + 2./3.*Pi + 2.*A

#V = np.array([[V11, V12],[V21, V22]])
#V.shape
V = np.zeros((Nr+1, 2, 2))
for i in range(Nr+1):
    V[i] = np.array([[V11[i], V12[i]],[V21[i], V22[i]]])


# start calculation
mu = 2.85 * convm # mass of system
E = 1000 * convE # collision energy

S, Y_N = numerov_asymptotic(E, 0, mu, V, r)

print('Log derivative matrix Y_N:\n', Y_N)
print('Scattering matrix S:\n', S)
print(np.sum(np.abs(S[:,0])**2), np.sum(np.abs(S[:,1])**2))


# calculation for different Nr
Nrvals = np.linspace(50, 400, 100, dtype=int)
convergence = np.zeros(len(Nrvals))
for iNr, Nr in enumerate(Nrvals):
    # setting up the grid:
    r = np.linspace(3.5, 22, Nr)
    h = r[2] - r[1]
    r = np.append(r, r[-1]+h)
    Sigma = pot(r, 0.813, 1.2014, 5.5701,
                3.7457e+3*convE, 6.7280e+5*convE, -1.2773e+5*convE, 3.4733e+6*convE)
    Pi = pot(r, 0.677, 2.2061, 6.2093,
             -7.7718e+3*convE, -7.3837e+7*convE, 3.2149e+7*convE, 2.9743e+6*convE)
    V11 = 2./3.*Sigma + Pi/3. - A
    V12 = np.sqrt(2.)/3.*(Pi - Sigma)
    V21 = V12
    V22 = Sigma/3. + 2./3.*Pi + 2.*A
    V = np.zeros((Nr+1, 2, 2))
    for i in range(Nr+1):
        V[i] = np.array([[V11[i], V12[i]],[V21[i], V22[i]]])
    S, Y_N = numerov_asymptotic(E, 0, mu, V, r)
    convergence[iNr] = np.abs(S[0,1])**k
print(convergence)


## calculation for different energies
#energies = np.linspace(100, 5000, 50)
#sigma = np.zeros(len(energies))
#for iE, E in enumerate(energies):
#    S, Y_N = numerov_asymptotic(E, 0, mu, V, r)
#    sigma[iE] = np.pi/(k*k) + np.abs(1. - S)**2


