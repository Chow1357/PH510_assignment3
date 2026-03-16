#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#------------------------------
# Boundary test
#------------------------------
def is_boundary(i, j, N):
    """
    function determines when the walker has
    reached the absorbing boundary
    """
    return i == 0 or i == N+1 or j == 0 or j == N+1
# One random walk test
def single_walk(start_i, start_j, N, rng):
    """
    performs a random walk from a start point on the grid (start_i, start_j)
    and creates an array and records how many times this walker visits each point.
    """
    visits = np.zeros((N+2, N+2), dtype=int)

    i, j = start_i, start_j

    while not is_boundary(i, j, N):
        visits[i, j] += 1
        # this matches 
        step = rng.integers(4) # random number generator with equal probability (directions) 
        if step == 0:
            i += 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j += 1
        else:
            j -= 1
    return visits

# monte carlo Greens's function with dynamic load balancing 
def estimate_greens_function(start_i. start_j, N, nwalkers, factor=0.25, seed=None, chunk_size=1000):
    """
    Estimate the Green's function using MPI with dynamic chunk scheduling
    """
    
    # random number generator, giving a different stream per rank
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed + rank)

    # local accumulators on each rank
    local_sum_visits = np.zeros((N + 2, N + 2), dtype=float)
    local_sumsq_visits = np.zeros((N + 2, N + 2), dtype=float)

    # special case: serial run
    if size == 1:
        for _ in range(nwalkers):
            visits = single_walk(start_i, start_j, N, rng)
            local_sum_visits += visits
            local_sumsq_visits += visits**2

    if rank == 0:
        next_walker = 0
        active _walkers = 0

        #send initial chunks to workers
        for worker in range (1, size):
            if next_walker < nwalkers:
                nchunk = min(chnk_size, nwalkers - next_walker)
                comm.send(nchunk, dest=worker, tag=1)
                next_walker += nchunk
                active_workers += 1
            else:
                # no work left 
                com.send(0, dest=worker, tag=1)

        # keep assigning chunks as workers finish
        while active_workers > 0:
            finished_worker = comm.recv(source=MPI.ANY_SOURCE, tag=2)

            if next_walker < nwalkers:
                nchunk = min(chunk_size, nwalkers - next_walker)
                comm.send(nchunk, dest=finished_worker, tag=1)
                next_walker += nchunk 
