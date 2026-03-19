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

