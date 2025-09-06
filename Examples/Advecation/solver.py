import numpy as np
import tomllib
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    # RHS for the advection equation
    dx = x[1] - x[0]
    dxu = grad(u[0], dx)

    dtu[0][:] = -dxu[:]
    dtu[0][0] = 0.0


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
    with open(filename, "w", encoding="utf-8") as f:
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

def make_animation(x, frames, times, y_min=None, y_max=None, out_file=None, dt_ms=50):
    """
    Build and (optionally) save a Matplotlib animation from stored frames.

    Parameters
    ----------
    x : array
        X coordinates (shared for all frames)
    frames : list of 1D arrays
        Sequence of u arrays for each frame
    times : list of float
        Simulation times corresponding to frames
    y_min, y_max : float, optional
        Fixed y-limits; if None, computed from data
    out_file : str, optional
        If provided, attempt to save animation to this file (mp4/gif)
    dt_ms : int
        Delay between frames in milliseconds
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], linewidth=2)

    if y_min is None or y_max is None:
        all_min = min(float(np.min(f)) for f in frames)
        all_max = max(float(np.max(f)) for f in frames)
    else:
        all_min, all_max = y_min, y_max

    # Avoid zero range
    y_range = max(all_max - all_min, 1e-12)
    pad = 0.1 * y_range
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(all_min - pad, all_max + pad)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Amplitude (u)')
    ax.grid(True)

    def init():
        line.set_data([], [])
        ax.set_title('Wave Animation (initializing)')
        return (line,)

    def update(i):
        line.set_data(x, frames[i])
        ax.set_title(f'Wave at t={times[i]:.2e}')
        return (line,)

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(frames), interval=dt_ms, blit=True
    )

    # Try to save if requested
    if out_file:
        try:
            if out_file.lower().endswith('.mp4'):
                anim.save(out_file, writer='ffmpeg', dpi=200)
            elif out_file.lower().endswith('.gif'):
                anim.save(out_file, writer='pillow', dpi=200)
            else:
                # Default to mp4 if extension unrecognized
                anim.save(out_file + '.mp4', writer='ffmpeg', dpi=200)
            print(f"Saved animation to {out_file}")
        except Exception as e:
            print(f"Could not save to {out_file} ({e}). Trying GIF fallback...")
            try:
                fallback = (out_file.rsplit('.', 1)[0] if '.' in out_file else out_file) + '.gif'
                anim.save(fallback, writer='pillow', dpi=200)
                print(f"Saved animation to {fallback}")
            except Exception as e2:
                print(f"Could not save animation ({e2}). Displaying live instead.")

    return anim, fig, ax


def main(parfile, visualize=False):
    """
    Main function to run the advection equation solver.

    Parameters:
    -----------
    parfile : str
        Path to the parameter file
    visualize : bool, optional
        Whether to create an animation at the end of the simulation (default: False)
    """
    # Read parameters
    with open(parfile, "r", encoding="utf-8") as f:
        params = tomllib.load(f)

    if visualize:
        frames = []
        times = []

    # Create the grid and set time step size
    x, dx = create_grid(params)
    dt = params["cfl"] * dx

    # Allocating memory
    u1 = np.empty_like(x)
    u = [u1]
    u_names = ["phi"]

    initial_data(u, x, params)

    if visualize:
        frames.append(u[0].copy())
        times.append(0.0)

    nt = params["nt"]
    time = 0.0

    iter = 0
    freq = params.get("output_frequency", 1)
    fname = f"data_{iter:04d}.curve"
    write_curve(fname, time, x, u_names, u)

    # Integrate in time
    for i in range(1, nt+1):
        rk2(u, x, dt)
        time += dt
        if i % freq == 0:
            print(f"Step {i:d}, t={time:.2e}, |u|={l2norm(u[0]):.2e}")
            fname = f"data_{i:04d}.curve"
            write_curve(fname, time, x, u_names, u)

            if visualize:
                frames.append(u[0].copy())
                times.append(time)

    if visualize and len(frames) > 1:
        # Use output_frequency and dt to set a sensible playback speed
        freq = params.get("output_frequency", 1)
        dt_ms = max(20, int(1000 * dt * freq))
        # Attempt to save an MP4; fallback to GIF if ffmpeg is not available
        out_file = params.get("animation_file", "wave.mp4")
        _, fig, _ = make_animation(x, frames, times, out_file=out_file, dt_ms=dt_ms)
        plt.show()




if __name__ == "__main__":
    # Parse command line arguments
    visualize = False  # Default: no visualization
    parfile = None

    # Check for visualization flag and parameter file
    for arg in sys.argv[1:]:
        if arg == "-v":
            # Enable visualization when -v flag is present
            visualize = True
        elif not arg.startswith("-"):
            # Argument without leading dash is treated as parameter file
            parfile = arg

    # Ensure a parameter file was provided
    if parfile is None:
        print("Usage: python solver.py [-v] <parfile>")
        print("  -v: generate and display an animation at the end of the simulation")
        sys.exit(1)
        
    # Run the main function with the parameter file and visualization flag
    main(parfile, visualize=visualize)
