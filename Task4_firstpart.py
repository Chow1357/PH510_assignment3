#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np
from mpi4py import MPI
from Task2_optimised import estimate_greens_function

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#converting physcal points to grid points

