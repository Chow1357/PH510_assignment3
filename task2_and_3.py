#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np
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

# monte carlo Green's function
def estimate_greens_function(start_i, start_j, N, nwalkers, factor=0.25, seed=None):
    """
    """
    #numpy random generator 
    rng = np.random.default_rng(seed + rank)
    # stores stats over many walkers (N+2 as gird includes halo)
    local_sum_visits = np.zeros((N+2, N+2), dtype=float)
    local_sumsq_visits = np.zeros((N+2, N+2), dtype=float)

    # dividing the walkers between the different processors
    base = nwalkers // size 
    remainder = nwalkers % size

    if rank < remainder:
        local_nwalkers = base + 1
    else:
        local_nwalkers = base 

    for _ in rnage(nwalkers):
        visits = single_walk(start_i, start_j, N, rng)
        # accumulates stats for each lattice site used for mean and variance later
        local_sum_visits += visits
        local_sumsq_visits += visits**2

    # arrays to hold combined sums from all processors
    gloabal_sum_visits = np.zeros((N+2, N+2), dtype=float)
    global_sumsq_visits = np.zeros((N+2, N+2), dtype=float)
 
    # combining the sums from all the processors using MPI 
    comm.Reduce(local_sum_visits, global_sum_visits, op=MPI.SUM, root=0)
    comm.Reduce(local_sumsq_visits, global_sumsq_visits, op=MPI.SUM, root=0)

    # computing the mean visits per walker for a point [i, j]
    if rank == 0:

        mean_visits = global_sum_visits / nwalkers
 
        # computing the variance
        if nwalkers >  1:
            var_visits = (global_sumsq_visits - nwalkers * mean_visits**2) / (nwalkers - 1)
            var_visits = np.maximum(var_visits, 0.0)
        else:
            var_visits = np.zeros_like(mean_visits)
        # standard eviation and standard error calculations
        std_visits = np.sqrt(var_visits)
        stderr_visits = std_visits / np.sqrt(nwalkers)
        # conversion of visits to Green's function with 0.25 factor
        G = factor * mean_visits
        G_std = factor * std_visits
        G_stderr = factor * stderr_visits 

        return G, G_std, G_stderr, mean_visits, std_visits

    return None, None, None, None, None

# main section of the program where we implement the grid parameters
#grid size N x N

if __name__ == "__main__":

    N = 50 

    # number of walkers across all MPI ranks
    nwalkers = 10000

    # test point near the centre 
    start_i = 25 
    start_j = 25 
    factor = 0.25 
    seed = 1234

    #calling the function to return the stated values
    G, G_std, G_stderr, mean_visits, std_visits = estimate_greens_function(
    start_i, start_j, N, nwalkers, factor=factor, seed=seed
    )
    # only root prints
    if rank == 0:

        print(f"Grid size (interior): {N} x {N}")
        print(f"starting point: ({start_i}, {start_j})")
        print(f"Number of walkers: {nwalkers}")
        print("Estimated Greens's function at the start point:", G[start_i, start_j])
        print(f"standard deviation at the start point: {G_std[start_i, start_j]:.6f}")
        print(f"Standard error at start point: {G_stderr[start_i, start_j]:.6f}")

    #--------------------------
    # TASK 3
    #--------------------------

    L = 100 

    # defining the different points to test on the grid in length scale
    points = [(50, 50), (2, 2), (2, 50)] # in cm 

    # function that converts a physical position in cm 
    # to a corresponding position on the grid
    def physical_to_grid(x_cm, y_cm, L_cm, N):
        """
        """

        h = L_cm / (N+1)
    
        i = int(round(x_cm / h))
        j = int(round(y_cm / h))

        i = int(round(x_cm / h))
        j = int(round(y_cm / h))

        return i, j

for x_cm, y_cm in points: 
 
    start_i, start_j = physical_to_grid(x_cm, y_cm, L, N)
 
    G, G_std, G_stderr, mean_visits, std_visits = estimate_greens_function(start_i, start_j, N, nwalkers, factor=0.25, seed=1234)

    print()
    print(f"Physical point: ({x_cm} cm, {y_cm} cm)")
    print(f"Grid index: ({start_i}, {start_j})")

    print("Green's function:", G[start_i, start_j])
    print("Standard deviation:", G_std[start_i, start_j])
    print("Standard error:", G_stderr[start_i, start_j])
    
    




