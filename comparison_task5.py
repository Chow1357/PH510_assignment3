#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np 
from mpi4py import MPI 

#importing functiosn from other scripts to save rewriting
from Task2_optimised import estimate_greens_function
from Task4 import (
    physical_to_grid,
    make_boundary_array,
    make_zero_charge,
    make_uniform_charge,
    make_gradient_charge,
    make_exponential_charge,
    potential_from_greens,
)
from part1 import solve_poisson_sor   # change filename if needed

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

