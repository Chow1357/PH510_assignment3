#!/opt/software/anaconda/python-3.10.9/bin/python
"""
this code is ran on python3.10.9
Monte carlo estimation of the descrete Green's function for the 2D poisson equation

This code implements random walks with absorbing boundaries and uses MPI with dynamic
load balancing to distribute walkers across processes

MIT License

Copyright (c) 2026 Chow1357

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#------------------------------
# Boundary test
#------------------------------
def is_boundary(i, j, N):
    """
    function determines whether the walker has
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
    return visits, i, j

# monte carlo Greens's function with dynamic load balancing
def estimate_greens_function(start_i, start_j, N, nwalkers, seed=None, chunk_size=2500):
    """
    Estimate the discrete charge-related Green's function using Monte carlo random walks.

    Each walker starts at (start_i, start_j), performs an unbiased random walk,
    and terminates upon reaching the boundary. The Green's function is estimated as:
        G = h^2 * <number of visits>
    Boundary-hit probabilities are also accumulated to represent the
    edge-potential (laplace) contributio
    """

    # random number generator, giving a different stream per rank
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed + rank)

    # local accumulators on each rank
    local_sum_visits = np.zeros((N + 2, N + 2), dtype=float)
    local_sumsq_visits = np.zeros((N + 2, N + 2), dtype=float)
    local_boundary_hits = np.zeros((N + 2, N + 2), dtype=float)

    # special case: serial run
    if size == 1:
        for _ in range(nwalkers):
            visits, bi, bj = single_walk(start_i, start_j, N, rng)
            local_sum_visits += visits
            local_sumsq_visits += visits**2
            local_boundary_hits[bi, bj] += 1
    else:
        if rank == 0:
            next_walker = 0
            active_workers = 0

            # send initial chunks to workers
            for worker in range(1, size):
                if next_walker < nwalkers:
                    nchunk = min(chunk_size, nwalkers - next_walker)
                    comm.send(nchunk, dest=worker, tag=1)
                    next_walker += nchunk
                    active_workers += 1
                else:
                    comm.send(0, dest=worker, tag=1)

            # rank 0 also works on chunks
            while True:

                # rank 0 takes a chunk for itself if any work remains
                if next_walker < nwalkers:
                    nchunk0 = min(chunk_size, nwalkers - next_walker)
                    next_walker += nchunk0

                    for _ in range(nchunk0):
                        visits, bi, bj = single_walk(start_i, start_j, N, rng)
                        local_sum_visits += visits
                        local_sumsq_visits += visits**2
                        local_boundary_hits[bi, bj] += 1
                else:
                    nchunk0 = 0

                # keep assigning chunks as workers finish (non- blocking check)
                while active_workers > 0 and comm.Iprobe(source=MPI.ANY_SOURCE, tag=2):
                    finished_worker = comm.recv(source=MPI.ANY_SOURCE, tag=2)

                    if next_walker < nwalkers:
                        nchunk = min(chunk_size, nwalkers - next_walker)
                        comm.send(nchunk, dest=finished_worker, tag=1)
                        next_walker += nchunk
                    else:
                        comm.send(0, dest=finished_worker, tag=1)
                        active_workers -= 1

                # stop when no work remains anywhere
                if next_walker >= nwalkers and active_workers == 0:
                    break

                # if rank 0 has no local work, wait for workers
                if nchunk0 == 0 and active_workers > 0:
                    finished_worker = comm.recv(source=MPI.ANY_SOURCE, tag=2)

                    if next_walker < nwalkers:
                        nchunk = min(chunk_size, nwalkers - next_walker)
                        comm.send(nchunk, dest=finished_worker, tag=1)
                        next_walker += nchunk
                    else:
                        comm.send(0, dest=finished_worker, tag=1)
                        active_workers -= 1

        else:
            # worker ranks repeatedly receive chunks
            while True:
                nchunk = comm.recv(source=0, tag=1)

                if nchunk == 0:
                    break

                for _ in range(nchunk):
                    visits, bi, bj = single_walk(start_i, start_j, N, rng)
                    local_sum_visits += visits
                    local_sumsq_visits += visits**2
                    local_boundary_hits[bi, bj] += 1
                # tell rank 0 this chunk is complete
                comm.send(rank, dest=0, tag=2)

    # global accumulators on rank 0
    global_sum_visits = np.zeros((N + 2, N + 2), dtype=float)
    global_sumsq_visits = np.zeros((N + 2, N + 2), dtype=float)
    global_boundary_hits = np.zeros((N + 2, N + 2), dtype=float)
    # combining the sums from all the processors using MPI
    comm.Reduce(local_sum_visits, global_sum_visits, op=MPI.SUM, root=0)
    comm.Reduce(local_sumsq_visits, global_sumsq_visits, op=MPI.SUM, root=0)
    comm.Reduce(local_boundary_hits, global_boundary_hits, op=MPI.SUM, root=0)
    # computing the mean visits per walker for a point [i, j]
    if rank == 0:
        mean_visits = global_sum_visits / nwalkers

        # computing the variance
        if nwalkers > 1:
            var_visits = (global_sumsq_visits - nwalkers * mean_visits**2) / (nwalkers - 1)
            var_visits = np.maximum(var_visits, 0.0)
        else:
            var_visits = np.zeros_like(mean_visits)

        # standard deviation and standard error calculations
        std_visits = np.sqrt(var_visits)
        stderr_visits = std_visits / np.sqrt(nwalkers)

        # conversion of visits to Green's function with h^2 factor
        h = 1.0 / (N+1)
        # convert expected visit counts into the discrete green's function
        G = h*h * mean_visits
        G_std = h*h * std_visits
        G_stderr = h*h * stderr_visits

        boundary_prob = global_boundary_hits / nwalkers

        return G, G_std, G_stderr, mean_visits, std_visits, boundary_prob

    return None, None, None, None, None, None
# main section of the program where we implement the grid parameters
#grid size N x N

if __name__ == "__main__":

    N = 50

    # number of walkers across all MPI ranks
    nwalkers = 200000

    # test point near the centre
    start_i = 25
    start_j = 25
    seed = 1234

    #calling the function to return the stated values
    G, G_std, G_stderr, mean_visits, std_visits, boundary_prob = estimate_greens_function(
    start_i, start_j, N, nwalkers, seed=seed, chunk_size=2500
    )
    # only root prints
    if rank == 0:

        print(f"Grid size (interior): {N} x {N}")
        print(f"starting point: ({start_i}, {start_j})")
        print(f"Number of walkers: {nwalkers}")
        print("Estimated Greens's function at the start point:", G[start_i, start_j])
        print(f"standard deviation at the start point: {G_std[start_i, start_j]:.6f}")
        print(f"Standard error at start point: {G_stderr[start_i, start_j]:.6f}")
        print(f"Sum of boundary probabilities: {np.sum(boundary_prob):.6f}")
