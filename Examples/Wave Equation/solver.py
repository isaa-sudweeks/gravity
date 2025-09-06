import numpy as np
import tomllib
import sys

def create_grid(params):
    x_min = params["x_min"]
    x_max = params["x_max"]
    nx = params["nx"]
    dx = (x_max - x_min) / (nx -1)
    x = np.zeros(nx)
    for i in range(nx):
        x[i] = x_min + i*dx
    return x, dx

def initial_data(u, x, params):
    x0 = params.get("id_x0", 0.5)
    amp = params.get("id_amp", 1.0)
    omega = params.get("id_omega", 1.0)
    u[0][:] = amp * np.exp(-omega * (x[:] - x0) ** 2)
    u[1][:] = 0.0

def grad(u, dx):
    du = np.zeros_like(u)
    idx_by_12 = 1.0 / (12 * dx)

    # center stencil
    du[2:-2] = (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[0:-4]) * idx_by_12

    # 4th order boundary stencils
    du[0] = (-25 * u[0] + 48 * u[1] - 36 * u[2] + 16 * u[3] - 3 * u[4]) * idx_by_12
    du[1] = (-3 * u[0] - 10 * u[1] + 18 * u[2] - 6 * u[3] + u[4]) * idx_by_12
    du[-2] = (-u[-5] + 6 * u[-4] - 18 * u[-3] + 10 * u[-2] + 3 * u[-1]) * idx_by_12
    du[-1] = (
            3 * u[-5] - 16 * u[-4] + 36 * u[-3] - 48 * u[-2] + 25 * u[-1]
        ) * idx_by_12

    return du


def rhs(dtu, u, x):
    # RHS for the wave  equation
    dx = x[1] - x[0]
    Phi = u[0]
    Pi = u[1]
    dxPhi = grad(Phi, dx)
    dxPi = grad(Pi, dx)

    dtu[0][:] = dxPi[:]
    dtu[1][:] = dxPhi[:]

    # left boundary conditions
    dtu[0][0] = dxPhi[0]
    dtu[1][0] = dxPi[0]

    # right boundary conditions
    dtu[0][-1] = -dxPhi[-1]
    dtu[1][-1] = -dxPi[-1]
    

def rk2(u, x, dt):
    nu = len(u)

    up = []
    k1 = []
    for i in range(nu):
        ux = np.empty_like(u[0])
        kx = np.empty_like(u[0])
        up.append(ux)
        k1.append(kx)

    rhs(k1, u, x)
    for i in range(nu):
        up[i][:] = u[i][:] + 0.5 * dt * k1[i][:]

    rhs(k1, up, x)
    for i in range(nu):
        u[i][:] = u[i][:] + dt * k1[i][:]

def write_curve(filename, time, x, u_names, u):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(u_names)):
            f.write(f"# {u_names[m]}\n")
            for xi, di in zip(x, u[m]):
                f.write(f"{xi:.8e} {di:.8e}\n")

def l2norm(u):
    """
    Compute the L2 norm of an array.
    """
    return np.sqrt(np.mean(u**2))


def main(parfile):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    # Create the grid and set time step size
    x, dx = create_grid(params)
    dt = params["cfl"] * dx

    # Allocating memory
    Phi = np.empty_like(x)
    Pi = np.empty_like(x)
    u = [Phi, Pi]
    u_names = ["Phi", "Pi"]

    initial_data(u, x, params)

    nt = params["nt"]
    time = 0.0

    iter = 0
    fname = f"data_{iter:04d}.curve"
    write_curve(fname, time, x, u_names, u)

    freq = params.get("output_frequency", 1)

    # Inegrate in time
    for i in range(1, nt+1):
        rk2(u, x, dt)
        time += dt
        if i % freq == 0:
            print(f"Step {i:d}, t={time:.2e}, |Phi|={l2norm(u[0]):.2e}, |Pi|={l2norm(u[1]):.2e}")
            fname = f"data_{i:04d}.curve"
            write_curve(fname, time, x, u_names, u)

    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)

    parfile = sys.argv[1]
    main(parfile)
