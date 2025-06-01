from matplotlib import pyplot as plt

import numpy as np
import math
from numpy import linalg
import time

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150

def jacobi_v1(ti, dt):
    """
    Jacobi method (Vector version)
    
    Parameters
    ----------
    n : integer
        size
    ti : float
        current time
        
    Parameters
    -----------
    dt : array
        difference
    """
    dt[1:-1, 1:-1, 1:-1] = (
        ti[:-2, 1:-1, 1:-1] + ti[1:-1, :-2, 1:-1] + ti[1:-1, 1:-1, :-2]
        + ti[2:, 1:-1, 1:-1] + ti[1:-1, 2:, 1:-1] + ti[1:-1, 1:-1, 2:]
        )/6 - ti[1:-1, 1:-1, 1:-1]
    
def bc(t):
    t[1:-1, -1, 1:-1] = 100   # Left
    t[1:-1, 0, 1:-1] = 100    # Right
    t[1:-1, 1:-1, -1] = 100   # Bottom
    t[1:-1, 1:-1, 0] = 300    # Top
    t[0, 1:-1, 1:-1] = 100    # Front
    t[-1, 1:-1, 1:-1] = 100   # Back

def Laplace_serial(n, tol):

    ti = np.zeros((n+2, n+2, n+2))
    dt = np.zeros_like(ti)
            
    err = 1
    hist_jacobi = []

    while err > tol:
        # Apply BC
        bc(ti)
        
        # Run Jacobi
        jacobi_v1(ti, dt)
        
        # Compute Error
        err = linalg.norm(dt) / n
        hist_jacobi.append(err)
        
        # Update solution
        ti += dt

    return ti, hist_jacobi


# Generate points (excluding BC)
n= 62
tol = 1e-4
xi = np.linspace(0, 1, n+2)
xx, yy, zz = np.meshgrid(xi[1:-1], xi[1:-1], xi[1:-1])
ti_serial, _ = Laplace_serial(n, tol)
np.save("ti_serial.npy", ti_serial)

# Save Computing time
ns = np.arange(30,79,8)
tol = 1e-4
times_serial =[]

for n in ns:
    t_s_serial = time.time()
    Laplace_serial(n, tol)
    t_e_serial = time.time()
    times_serial.append(t_e_serial - t_s_serial)

np.save("time_serial.npy", times_serial)
