#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np 

# defining the grid on which we can solve poisson's equation
N = 50

# target value for change in solution
target = 1e-8

# defining the boundary conditions and introducing a potential difference
V_top = 100.0 
V_bottom = 100.0
V_left = 0.0
V_right = 0.0

# grid spacing 
h = 1.0 / (N - 1) 

# SOR parameter for square grid 
omega = 2.0 / (1.0 + np.sin(np.pi / N))

# potential array with halo shape 
phi = np.zeros([N + 2, N + 2], dtype=float)

# source term for poisson 
f = np.zeros([N + 2, N + 2], dtype=float)

# example: positive and negative source
f[N // 4 + 1, N // 4 + 1] = 100.0
f[3 * N //4 + 1, 3 * N // 4 + 1] = -100.0

# setting the physical boundaries where PDE is solved
phi[1, 1:N+1] = V_top
phi[N, 1:N+1] = V_bottom 
phi[1:N+1, 1] = V_left 
phi[1:N+1, N] = V_right

# filling the halo cells by copying the boundary values outward
phi[0, 1:N+1] = phi[1, 1:N+1] 
phi[N+1, 1:N+1] = phi[N, 1:N+1]
phi[1:N+1, 0] = phi[1:N+1, 1]
phi[1:N+1, N+1] = phi[1:N+1, N]

# corners of the halo
phi[0, 0] = phi[1, 1]
phi[0, N+1] = phi[1, N]
phi[N+1, 0] = phi[N, 1]
phi[N+1, N+1] = phi[N, N]

# defining the poisson over relaxation method
def poisson_sor(phi, f):
    max_change = 0.0
    
    for i in range(2, N):
        for j in range(2, N):
            old_value = phi[i, j]

            # Poisson update from the neighbours and the source term f 
            phi_star = 0.25 * (
                phi[i + 1, j] + phi[i - 1, j] +
                phi[i, j + 1] + phi[i, j - 1] -
                h * h * f[i, j]
            )
            # SOR update 
            phi[i, j] = (1.0 - omega) * old_value + omega * phi_star

            change = abs(phi[i, j] - old_value)
            if change > max_change:
                max_change = change
    return max_change

#starting the loop which repeats the SOR sweep until convergence is acheived
#i.e tolerance is exceeded
delta = 1.0 
iterations = 0 

while delta > target: 
    delta = poisson_sor(phi, f)
    iterations += 1

#print functions for important parameters 
print("Iterations =", iterations) 
print("Omega =", omega) 
print("Final max change =", delta) 






