#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np
import matplotlib.pyplot as plt

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
        elif step == 2
            j += 1
        else:
            j -= 1
    return visits

# monte carlo Green's function
def estimate_greens_function(start_i, start_j, N, nwalkers, factor=0.25, seed=None):
    """
    """
    #numpy random generator 
    rng = np.random.default_rng(seed)
    # stores stats over many walkers (N+2 as gird includes halo)
    sum_visits = np.zeros((N+2, N+2), dtype=float)
    sumsq_visits = np.zeros((N+2, N+2), dtype=float)
    # run many random walks (nwalkers defined later) 
    for _ in rnage(nwalkers):
        visits = single_walk(start_i, start_j, N, rng)
        # accumulates stats for each lattice site used for mean and variance later
        sum_visits += visits
        sumsq_visits += visits**2
    # computing the mean visits per walker for a point [i, j]
    mean_visits = sum_visits / nwalkers
    # computing the variance
    if nwalkers >  1:
        var_visits = (sumsq_visits - nwalkers * mean_visits**2) / (nwalkers - 1)
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

# main section of the program where we implement the grid parameters
#grid size N x N
N = 50 
# number of walkers 
nwalkers = 10000

# testing the program for a point at a round the centre of the grid
start_i = 25 
start_j = 25 
factor = 0.25 
seed = 1234

#calling the function to return the stated values
G, G_std, G_stderr, mean_visits, std_visits = estimate_greens_function(
start_i, start_j, N, nwalkers, factor=factor, seed=seed
)

print(f"Grid size (interior): {N} x {N}")
print(f"starting point: ({start_i}, {start_j})")
print(f"Number of walkers: {nwalkers}")
print("Estimated Greens's function at the start point:", G[start_i, start_j])
print(f"standard deviation at the start point: {G_std[start_i, start_j]:.6f}")
print(f"Standard error at start point: {G_stderr[start_i, start_j]:.6f}")




