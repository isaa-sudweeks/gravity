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

def visualize_wave(x, u, width=80, height=20):
    """
    Create a simple ASCII visualization of the wave.
    
    This function creates a text-based visualization of the wave in the terminal
    using ASCII characters. It maps the wave amplitude to a grid of characters,
    with '*' representing the wave. The visualization is scaled to fit within
    the specified width and height.
    
    Parameters:
    -----------
    x : array
        The x coordinates
    u : array
        The wave amplitude at each x
    width : int
        The width of the visualization in characters
    height : int
        The height of the visualization in characters
    """
    # Find min and max values for scaling
    u_min = np.min(u)
    u_max = np.max(u)
    
    # Avoid division by zero
    if u_max == u_min:
        u_max = u_min + 1.0
    
    # Create a 2D grid for the visualization
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Fill the grid with the wave
    for i, val in enumerate(u):
        # Map x to column
        col = int((i / len(u)) * width)
        if col >= width:
            col = width - 1
            
        # Map u to row (invert so higher values are at the top)
        row = int((1.0 - (val - u_min) / (u_max - u_min)) * (height - 1))
        if row >= height:
            row = height - 1
        if row < 0:
            row = 0
            
        # Place a character at the position
        grid[row][col] = '*'
    
    # Print the visualization
    print("\nWave Visualization:")
    print("-" * width)
    for row in grid:
        print(''.join(row))
    print("-" * width)
    print(f"Min: {u_min:.4f}, Max: {u_max:.4f}")


def main(parfile, visualize=False):
    """
    Main function to run the wave equation solver.
    
    Parameters:
    -----------
    parfile : str
        Path to the parameter file
    visualize : bool, optional
        Whether to visualize the wave during simulation (default: False)
    """
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
    
    # Visualize initial state if requested
    if visualize:
        # Display ASCII visualization of both wave components
        print("\nVisualizing Phi (wave amplitude):")
        visualize_wave(x, u[0])
        print("\nVisualizing Pi (wave momentum):")
        visualize_wave(x, u[1])

    freq = params.get("output_frequency", 1)

    # Inegrate in time
    for i in range(1, nt+1):
        rk2(u, x, dt)
        time += dt
        if i % freq == 0:
            print(f"Step {i:d}, t={time:.2e}, |Phi|={l2norm(u[0]):.2e}, |Pi|={l2norm(u[1]):.2e}")
            fname = f"data_{i:04d}.curve"
            write_curve(fname, time, x, u_names, u)
            
            # Visualize both wave components at this time step if requested
            if visualize:
                print("\nVisualizing Phi (wave amplitude):")
                visualize_wave(x, u[0])
                print("\nVisualizing Pi (wave momentum):")
                visualize_wave(x, u[1])

    


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
        print("  -v: visualize the wave simulation")
        sys.exit(1)
        
    # Run the main function with the parameter file and visualization flag
    main(parfile, visualize=visualize)
