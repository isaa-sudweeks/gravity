import numpy as np
import gravity.rational as gr
import tomllib
import sys

def create_grid(params):
    x_min = gr.Rational()
    x_min = x_min.from_float(float(params["x_min"]))
    x_max = gr.Rational()
    x_max = x_max.from_float(params["x_max"])
    nx = gr.Rational()
    nx = nx.from_float(params["nx"])
    dx = (x_max - x_min) / (nx - 1)
    x_values = [gr.Rational(0, 1) for _ in range(int(nx))]
    for i in range(int(nx)):
        x_values[i] = x_min + i * dx
    x = gr.as_rational_array(x_values)
    return x, dx

def initial_data(u, x, params):
    x0 = gr.Rational()
    x0 = x0.from_float(float(params.get("id_x0", 0.5)))
    amp = gr.Rational()
    amp = amp.from_float(float(params.get("id_amp", 1.0)))
    omega = gr.Rational()
    omega = omega.from_float(float(params.get("id_omega", 1.0)))
    u[0][:] = amp * np.exp(-omega * (x[:] - x0) ** 2)
    u[1][:] = gr.zeros_like(u[1])


def grad(u, dx):
    values = gr.as_rational_array(u, copy=False)
    du = gr.zeros_like(values)
    idx_by_12 = gr.Rational(1, 12) / dx

    # Center Stencil
    du[2:-2] = (-values[4:] + 8 * values[3:-1] - 8 * values[1:-3] + values[0:-4]) * idx_by_12

    # 4th order boundary stencils
    du[0] = (-25 * values[0] + 48 * values[1] - 36 * values[2] + 16 * values[3] - 3 * values[4]) * idx_by_12
    du[1] = (-3 * values[0] - 10 * values[1] + 18 * values[2] - 6 * values[3] + values[4]) * idx_by_12
    du[-2] = (-values[-5] + 6 * values[-4] - 18 * values[-3] + 10 * values[-2] + 3 * values[-1]) * idx_by_12
    du[-1] = (3 * values[-5] - 16 * values[-4] + 36 * values[-3] - 48 * values[-2] + 25 * values[-1]) * idx_by_12
    return du

def rhs(dtu, u, x):
    # RHS for the wave equation
    dx = x[1] - x[0]
    Phi = u[0]
    Pi = u[1]
    dxPhi = grad(Phi, dx)
    dxPi = grad(Pi, dx)

    # left boundary condition
    dtu[0][0] = dxPi[0]
    dtu[1][0] = dxPhi[0]

    # Right boundary condition
    dtu[0][-1] = -dxPhi[-1]
    dtu[1][-1] = -dxPi[-1]

def rk2(u, x, dt):
    nu = len(u)

    up = [gr.zeros_like(u_field) for u_field in u]
    k1 = [gr.zeros_like(u_field) for u_field in u]

    rhs(k1, u, x)
    for i in range(nu):
        up[i][:] = u[i][:] + gr.Rational(1, 2) * dt * k1[i][:]

    rhs(k1, up, x)
    for i in range(nu):
        u[i][:] = u[i][:] + dt * k1[i][:]

def write_curve(filename, time, x, u_names, u):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(u_names)):
            f.write(f"# {u_names[m]}\n")
            for xi, di in zip(x, u[m]):
                f.write(f"{xi} {di}\n")

def l2norm(u):
    return np.sqrt(np.mean(u**2))

def main(parfile, output_path):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    x, dx = create_grid(params)
    dt = gr.Rational()
    dt = dt.from_float(float(params["cfl"]) * float(dx))

    Phi = gr.zeros(len(x))
    Pi = gr.zeros(len(x))
    u = [Phi, Pi]
    u_names = ["Phi", "Pi"]

    initial_data(u, x, params)

    nt = gr.Rational()
    nt = nt.from_float(params["nt"])
    time = gr.Rational(0, 1)

    iter = 0
    fname = f"{output_path}data_{iter:04d}.curve"
    write_curve(fname, float(time), np.float64(x), u_names, np.float64(u))

    freq = params.get("output_frequency", 1)

    for i in range(1, int(nt)+1):
        rk2(u, x, dt)
        time += dt
        if i % freq == 0:
            print(f"Step {i:d}, t={time}, |Phi|={l2norm(u[0]):.2e}, |Pi|={l2norm(u[1]):.2e}")
            fname = f"{output_path}data_{i:04d}.curve"

            write_curve(fname, float(time), np.float64(x), u_names, np.float64(u))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:  python solver.py <parfile> <output_path>")
        sys.exit(1)

    parfile = sys.argv[1]
    output_path = sys.argv[2]
    main(parfile, output_path)
