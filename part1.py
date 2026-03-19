#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_boundary_array(N, V_top, V_bottom, V_left, V_right)
    """
    create boundary values on a halo grid.
    Interior poinbts are 1..N and boundaries are 0 and N+1
    """
    
    phi = np.zeros((N + 2, N + 2), dtype=float)

    phi[0, :] = V_bottom
    phi[N + 1, :] = V_top
    phi[:, 0] = V_left
    phi[:, N +1] = V_right

    #corner values
    phi[0, 0] = 0.5 * (V_bottom + V_left)
    phi[0, N + 1] = 0.5 * (V_bottom + V_right)
    phi[N + 1, 0] = 0.5 * (V_top + V_left)
    phi[N + 1, N + 1] = 0.5 * (V_top + V_right)

    return phi 

# defining the poisson over relaxation method
def poisson_sor_step(phi, f, N, h, omega):
    max_change = 0.0
    
    for i in range(1, N + 1):
        for j in range(1, N + 1):
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

def solve_poisson_sor(N, f, V_top, V_bottom, V_left, V_right, target=1e-8):
    """
    Solve Poisson's equation on a halo grid using SOR.
    """
    # grid spacing 
    h = 1.0 / (N - 1) 
    # SOR parameter for square grid 
    omega = 2.0 / (1.0 + np.sin(np.pi / N))

    # potential array  
    phi = make_boundary_array(N, V_top, V_bottom, V_left, V_right)
    #starting the loop which repeats the SOR sweep until convergence is      acheived
    #i.e tolerance is exceeded
    delta = 1.0 
    iterations = 0  
 
    while delta > target:
        delta = poisson_sor_step(phi, f, N, h, omega)
        iterations += 1

    return phi, iterations, omega, delta

def make_example_charge(N):
    """
    Example source term for standalone testing.
    """
    # source term for poisson 
    f = np.zeros([N + 2, N + 2], dtype=float)

    # example: positive and negative source
    f[N // 4, N // 4] = 100.0
    f[3 * N //4, 3 * N // 4] = -100.0

    return f


if __name__ == "__main__"

    # defining the grid on which we can solve poisson's equation
    N = 50

    # target value for change in solution
    target = 1e-8

    # defining the boundary conditions and introducing a potential difference
    V_top = 100.0 
    V_bottom = 0.0
    V_left = 0.0
    V_right = 0.0
    
    f = make_example_charge(N)

    phi, iterations, omega, delta = solve_poisson_sor(
        N, f, V_top, V_bottom, V_left, V_right, V_right, target=target
    )

#print functions for important parameters 
print("Iterations =", iterations) 
print("Omega =", omega) 
print("Final max change =", delta) 

# plotting phi 
plt.figure()
plt.imshow(phi, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Potential (phi)')
plt.title('Solution of Poisson Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("phi.png", dpi=300)
plt.show()





