#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np
from mpi4py import MPI
from Task2_optimised import estimate_greens_function

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#converting physcal points to grid points
def physical_to_grid(x_cm, y_cm, L_cm, N):
    h = L_cm / (N + 1)
    i = int(round(y_cm / h))
    j = int(round(x_cm / h))
    i = max(1, min(N, i))
    j = max(1, min(N, j))
    return i, j

# building a new boundary potential array 
def make_boundary_array(N, V_top, V_bottom, V_left, V_right):
    B = np.zeros((N + 2, N + 2), dtype=float)

    B[0, :] = V_bottom
    B[N + 1, :] = V_top
    B[:, 0] = V_left
    B[:, N + 1] = V_right

    B[0, 0] = 0.5 * (V_bottom + V_left)
    B[0, N + 1] = 0.5 * (V_bottom + V_right)
    B[N + 1, 0] = 0.5 * (V_top + V_left)
    B[N + 1, N + 1] = 0.5 * (V_top + V_right)

    return B
# Zero-charge array 
#creating an interior charge-density array with zero everywhere
def make_zero_charge(N):
    return np.zeros((N + 2, N + 2), dtype=float)

# Turnign Greens functions into a potential 

