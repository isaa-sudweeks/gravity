import numpy as np
import tomllib
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

try:
    import torch
except Exception:
    torch = None

def _is_tensor(x):
    return (torch is not None) and torch.is_tensor(x)

def _zeros_like(x):
    if _is_tensor(x):
        return torch.zeros_like(x)
    return np.zeros_like(x)

def _empty_like(x):
    if _is_tensor(x):
        return torch.empty_like(x)
    return np.empty_like(x)

def _asarray_like(x, like):
    if _is_tensor(like):
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return np.asarray(x)

def _select_device(preferred=None):
    """
    Choose an appropriate torch device string.
    Prefer CUDA; else prefer MPS on Apple Silicon; else CPU.
    """
    if torch is None:
        return None
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _sync(device):
    if (torch is None) or (device is None):
        return
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass

# Prepare MPS for better stability and memory management
def _prepare_mps():
    """
    Prepare MPS backend to be less crashy on memory by lowering the high-watermark ratio
    and clearing caches before large allocations.
    """
    if (torch is None) or (not hasattr(torch.backends, "mps")) or (not torch.backends.mps.is_available()):
        return
    # Lower the MPS caching watermark unless the user has already set it
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
    try:
        torch.mps.empty_cache()
    except Exception:
        pass

# In-place 4th order gradient helper (NumPy or Torch)
def grad_inplace(u, dx, out):
    """
    Compute du/dx into `out` in-place using 4th-order centered differences
    with 4th-order one-sided stencils at the boundaries.
    Works with NumPy arrays or Torch tensors.
    """
    idx_by_12 = 1.0 / (12.0 * dx)
    out[2:-2] = (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[0:-4]) * idx_by_12
    out[0]  = (-25 * u[0] + 48 * u[1] - 36 * u[2] + 16 * u[3] - 3 * u[4]) * idx_by_12
    out[1]  = ( -3 * u[0] - 10 * u[1] + 18 * u[2] -  6 * u[3] +     u[4]) * idx_by_12
    out[-2] = ( -u[-5] +  6 * u[-4] - 18 * u[-3] + 10 * u[-2] +  3 * u[-1]) * idx_by_12
    out[-1] = (  3 * u[-5] - 16 * u[-4] + 36 * u[-3] - 48 * u[-2] + 25 * u[-1]) * idx_by_12

def create_grid(params, use_torch=False, device=None, dtype=None):
    x_min = params["x_min"]
    x_max = params["x_max"]
    nx = params["nx"]
    dx = (x_max - x_min) / (nx - 1)
    if use_torch and (torch is not None):
        # Always enforce float32 on Torch (best support/perf on GPUs, esp. MPS)
        dtype = torch.float32
        if device is None:
            device = _select_device()
        # If we're on MPS, clear cache before a large allocation
        if device is not None and getattr(device, "type", "") == "mps":
            _prepare_mps()
        # Try allocating on the selected device; if MPS (or any) fails, fall back to CPU Torch
        try:
            x = torch.linspace(x_min, x_max, steps=nx, device=device, dtype=dtype)
        except Exception as e:
            print(f"[warn] Torch allocation on {device} failed ({e}); falling back to torch CPU.")
            device = torch.device("cpu")
            x = torch.linspace(x_min, x_max, steps=nx, device=device, dtype=dtype)
    else:
        x = np.linspace(x_min, x_max, num=nx, dtype=float)
    return x, dx

def initial_data(u, x, params):
    x0 = params.get("id_x0", 0.5)
    amp = params.get("id_amp", 1.0)
    omega = params.get("id_omega", 1.0)
    if _is_tensor(u[0]):
        u[0][:] = amp * torch.exp(-omega * (x[:] - x0) ** 2)
    else:
        u[0][:] = amp * np.exp(-omega * (x[:] - x0) ** 2)

def grad(u, dx):
    # This is a finite difference using a centered fourth-order accurate stencil
    # and 4th-order accurate one-sided stencils at the boundaries
    du = _zeros_like(u)
    if _is_tensor(u):
        idx_by_12 = 1.0 / (12.0 * dx)
    else:
        idx_by_12 = 1.0 / (12.0 * dx)

    # center stencil
    du[2:-2] = (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[0:-4]) * idx_by_12

    # 4th order boundary stencils
    du[0] = (-25 * u[0] + 48 * u[1] - 36 * u[2] + 16 * u[3] - 3 * u[4]) * idx_by_12
    du[1] = (-3 * u[0] - 10 * u[1] + 18 * u[2] - 6 * u[3] + u[4]) * idx_by_12
    du[-2] = (-u[-5] + 6 * u[-4] - 18 * u[-3] + 10 * u[-2] + 3 * u[-1]) * idx_by_12
    du[-1] = (3 * u[-5] - 16 * u[-4] + 36 * u[-3] - 48 * u[-2] + 25 * u[-1]) * idx_by_12

    return du


def rhs(dtu, u, dx, tmp):
    """
    RHS for the advection equation u_t = -u_x.
    Uses `tmp` as scratch to hold the gradient.
    """
    grad_inplace(u[0], dx, tmp)
    dtu[0][:] = -tmp[:]
    dtu[0][0] = 0.0


def rk2(u, dt, dx, up, k1, tmp):
    """
    2-stage RK using preallocated buffers:
    - up: list of tmp states
    - k1: list of RHS buffers
    - tmp: scratch for spatial derivative
    """
    nu = len(u)
    rhs(k1, u, dx, tmp)
    for i in range(nu):
        up[i][:] = u[i][:] + 0.5 * dt * k1[i][:]
    rhs(k1, up, dx, tmp)
    for i in range(nu):
        u[i][:] = u[i][:] + dt * k1[i][:]

def write_curve(filename, time, x, u_names, u):
    # Convert to NumPy if tensors (for safe text I/O)
    x_np = x.detach().cpu().numpy() if _is_tensor(x) else x
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(u_names)):
            f.write(f"# {u_names[m]}\n")
            um = u[m].detach().cpu().numpy() if _is_tensor(u[m]) else u[m]
            for xi, di in zip(x_np, um):
                f.write(f"{xi:.8e} {di:.8e}\n")

def l2norm(u):
    """
    Compute the L2 norm of an array (NumPy or Torch).
    """
    if _is_tensor(u):
        return torch.sqrt(torch.mean(u * u)).item()
    else:
        return float(np.sqrt(np.mean(u**2)))

def solve_once(params, use_torch=False, device=None, do_write=False):
    """
    Run the solver once (no animation) and optionally skip disk I/O for fair benchmarking.
    Returns final time, final L2 norm, and elapsed wall time in seconds.
    """
    # Grid and dt
    x, dx = create_grid(params, use_torch=use_torch, device=device, dtype=(torch.float32 if (use_torch and torch is not None) else None))
    dt = params["cfl"] * dx

    # State
    u1 = _empty_like(x)
    u = [u1]
    u_names = ["phi"]
    initial_data(u, x, params)

    # Preallocate buffers for RK2
    tmp = _empty_like(u[0])
    up  = [ _empty_like(u[0]) ]
    k1  = [ _empty_like(u[0]) ]

    nt = params["nt"]
    time_t = 0.0

    # Optional warmup for torch to avoid first-call overhead in timing
    if use_torch and (torch is not None):
        with torch.no_grad():
            for _ in range(3):
                rk2(u, dt, dx, up, k1, tmp)

    # Begin timing
    t0 = time.perf_counter()
    if use_torch and (torch is not None):
        _sync(device)
        with torch.no_grad():
            for i in range(1, nt + 1):
                rk2(u, dt, dx, up, k1, tmp)
                time_t += dt
                if do_write and (i % params.get("output_frequency", 1) == 0):
                    fname = f"bench_{i:04d}.curve"
                    write_curve(fname, time_t, x, u_names, u)
        _sync(device)
    else:
        for i in range(1, nt + 1):
            rk2(u, dt, dx, up, k1, tmp)
            time_t += dt
            if do_write and (i % params.get("output_frequency", 1) == 0):
                fname = f"bench_{i:04d}.curve"
                write_curve(fname, time_t, x, u_names, u)
    t1 = time.perf_counter()

    l2 = l2norm(u[0])
    return time_t, l2, (t1 - t0)

def benchmark(parfile, device_str=None, repeats=3, do_write=False):
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    def _solve_get_final_numpy(p, use_torch=False, device=None):
        xg, dxg = create_grid(p, use_torch=use_torch, device=device, dtype=(torch.float32 if (use_torch and torch is not None and use_torch) else None))
        dtg = p["cfl"] * dxg
        u1g = _empty_like(xg); ug = [u1g]
        initial_data(ug, xg, p)
        tmpg = _empty_like(ug[0]); upg=[_empty_like(ug[0])]; k1g=[_empty_like(ug[0])]
        ntg = p["nt"]
        if use_torch and (torch is not None):
            with torch.no_grad():
                for _ in range(ntg):
                    rk2(ug, dtg, dxg, upg, k1g, tmpg)
            return ug[0].detach().cpu().numpy()
        else:
            for _ in range(ntg):
                rk2(ug, dtg, dxg, upg, k1g, tmpg)
            return ug[0].copy()
    u_cpu_final = _solve_get_final_numpy(params, use_torch=False, device=None)

    # NumPy CPU
    numpy_times = []
    for _ in range(repeats):
        _, l2_cpu, dt_cpu = solve_once(params, use_torch=False, device=None, do_write=do_write)
        numpy_times.append(dt_cpu)

    # Torch GPU (if available)
    torch_available = (torch is not None) and (torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    results = {}
    results["numpy_cpu_sec"] = (sum(numpy_times) / len(numpy_times))

    if torch_available:
        device = _select_device(device_str or params.get("device", None))
        if getattr(device, "type", "") == "mps":
            _prepare_mps()
        torch_times = []
        gpu_failed = False
        for _ in range(repeats):
            try:
                _, l2_gpu, dt_gpu = solve_once(params, use_torch=True, device=device, do_write=do_write)
                torch_times.append(dt_gpu)
            except Exception as e:
                print(f"[warn] GPU benchmark run failed on {device}: {e}")
                gpu_failed = True
                break
        results["torch_device"] = str(device)
        if (not gpu_failed) and len(torch_times) > 0:
            results["torch_gpu_sec"] = (sum(torch_times) / len(torch_times))
            results["speedup_x"] = results["numpy_cpu_sec"] / results["torch_gpu_sec"] if results["torch_gpu_sec"] > 0 else float("inf")
            # Accuracy vs CPU reference
            try:
                u_gpu_final = _solve_get_final_numpy(params, use_torch=True, device=device)
                denom = np.linalg.norm(u_cpu_final.ravel())
                rel_l2 = np.linalg.norm((u_gpu_final - u_cpu_final).ravel()) / (denom if denom > 0 else 1.0)
                max_abs = float(np.max(np.abs(u_gpu_final - u_cpu_final)))
                results["rel_l2_vs_cpu"] = float(rel_l2)
                results["max_abs_err_vs_cpu"] = max_abs
            except Exception as e:
                print(f"[warn] Accuracy check on GPU failed: {e}")
                results["rel_l2_vs_cpu"] = None
                results["max_abs_err_vs_cpu"] = None
        else:
            print("[warn] Skipping GPU timing/accuracy due to earlier failure.")
            results["torch_gpu_sec"] = None
            results["speedup_x"] = None
            results["rel_l2_vs_cpu"] = None
            results["max_abs_err_vs_cpu"] = None
    else:
        results["torch_device"] = None
        results["torch_gpu_sec"] = None
        results["speedup_x"] = None

    print("=== Benchmark Results ===")
    print(f"Steps (nt): {params['nt']}, Grid (nx): {params['nx']}")
    print(f"NumPy CPU avg time: {results['numpy_cpu_sec']:.6f} s")
    if results["torch_device"] is not None:
        print(f"Torch device: {results['torch_device']}")
        print(f"Torch GPU avg time: {results['torch_gpu_sec']:.6f} s")
        print(f"Speedup (CPU / GPU): {results['speedup_x']:.2f}×")
        print(f"Relative L2 error (GPU vs CPU): {results['rel_l2_vs_cpu']:.3e}")
        print(f"Max abs error   (GPU vs CPU): {results['max_abs_err_vs_cpu']:.3e}")
    else:
        print("Torch GPU not available; skipped GPU run.")
    return results

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
        xi = x.detach().cpu().numpy() if _is_tensor(x) else x
        fi = frames[i].detach().cpu().numpy() if _is_tensor(frames[i]) else frames[i]
        line.set_data(xi, fi)
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


def main(parfile, visualize=False, use_torch=False, device_str=None, cli_dtype=None, cli_compile=False):
    """
    Main function to run the advection equation solver.

    Parameters:
    -----------
    parfile : str
        Path to the parameter file
    visualize : bool, optional
        Whether to create an animation at the end of the simulation (default: False)
    use_torch : bool, optional
        Whether to use PyTorch backend (default: False)
    device_str : str or None, optional
        Device string override (e.g., "cuda", "mps", "cpu")
    cli_dtype : str or None
        CLI override for dtype (float32/float64)
    cli_compile : bool
        CLI flag to enable torch.compile (CUDA)
    """
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    if use_torch:
        if torch is None:
            print("PyTorch is not installed; falling back to NumPy.")
            use_torch = False
            device = None
            dtype = None
        else:
            # Allow device override via CLI or TOML
            dev_pref = device_str or params.get("device", None)
            device = _select_device(dev_pref)
            print(f"Using device: {device}")
            # dtype preference (Torch only)
            dtype_pref = (cli_dtype or params.get("dtype", None))
            if dtype_pref and str(dtype_pref).lower() == "float64":
                dtype = torch.float64
            else:
                dtype = torch.float32
            # Force FP32 on Torch (especially important for Apple MPS)
            if torch is not None:
                try:
                    torch.set_default_dtype(torch.float32)
                except Exception:
                    pass
            dtype = torch.float32
            if str(device) == "mps":
                _prepare_mps()
            # compile preference
            compile_flag = bool(cli_compile or params.get("compile", False))
    else:
        device = None
        dtype = None
        compile_flag = False

    if visualize:
        frames = []
        times = []

    # Create the grid and set time step size
    x, dx = create_grid(params, use_torch=use_torch, device=device, dtype=(dtype if use_torch else None))
    dt = params["cfl"] * dx

    # Allocating memory
    u1 = _empty_like(x)
    u = [u1]
    u_names = ["phi"]

    initial_data(u, x, params)

    # Preallocate buffers for RK2
    tmp = _empty_like(u[0])
    up  = [ _empty_like(u[0]) ]
    k1  = [ _empty_like(u[0]) ]

    # Optionally create a compiled RK2 wrapper (CUDA-optimized)
    rk2_compiled = None
    if use_torch and hasattr(torch, "compile") and compile_flag and str(device).startswith("cuda"):
        try:
            def rk2_wrap(u0, dt0, dx0, up0, k10, tmp0):
                rk2(u0, dt0, dx0, up0, k10, tmp0)
                return u0[0]
            rk2_compiled = torch.compile(rk2_wrap, mode="max-autotune")
            print("torch.compile enabled for rk2.")
        except Exception as _e:
            rk2_compiled = None
            print(f"torch.compile not used ({_e}).")

    if visualize:
        if _is_tensor(u[0]):
            frames.append(u[0].detach().cpu().numpy().copy())
        else:
            frames.append(u[0].copy())
        times.append(0.0)

    nt = params["nt"]
    timev = 0.0

    iter = 0
    freq = params.get("output_frequency", 1)
    fname = f"data_{iter:04d}.curve"
    write_curve(fname, timev, x, u_names, u)

    # Integrate in time
    for i in range(1, nt+1):
        if rk2_compiled is not None and _is_tensor(u[0]):
            with torch.no_grad():
                rk2_compiled(u, dt, dx, up, k1, tmp)
        else:
            if _is_tensor(u[0]):
                with torch.no_grad():
                    rk2(u, dt, dx, up, k1, tmp)
            else:
                rk2(u, dt, dx, up, k1, tmp)
        timev += dt
        if i % freq == 0:
            print(f"Step {i:d}, t={timev:.2e}, |u|={l2norm(u[0]):.2e}")
            fname = f"data_{i:04d}.curve"
            write_curve(fname, timev, x, u_names, u)

            if visualize:
                if _is_tensor(u[0]):
                    frames.append(u[0].detach().cpu().numpy().copy())
                else:
                    frames.append(u[0].copy())
                times.append(timev)

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
    use_torch = False
    device_str = None
    parfile = None
    run_bench = False
    bench_repeats = 3
    _cli_dtype = None
    _cli_compile = False

    # Check for visualization flag and parameter file
    for arg in sys.argv[1:]:
        if arg == "-v":
            # Enable visualization when -v flag is present
            visualize = True
        elif arg == "-t":
            use_torch = True
        elif arg in ("-b", "--bench"):
            run_bench = True
        elif arg.startswith("--repeats="):
            bench_repeats = int(arg.split("=", 1)[1])
        elif arg.startswith("--device="):
            device_str = arg.split("=", 1)[1]
        elif arg.startswith("--dtype="):
            _cli_dtype = arg.split("=", 1)[1].lower()
        elif arg == "--compile":
            _cli_compile = True
        elif not arg.startswith("-"):
            # Argument without leading dash is treated as parameter file
            parfile = arg

    # Ensure a parameter file was provided
    if parfile is None:
        print("Usage: python solver.py [-v] [-t] [-b] [--repeats=N] [--device=cuda|mps|cpu] [--dtype=float32|float64] [--compile] <parfile>")
        print("  -v: generate and display an animation at the end of the simulation")
        print("  -t: enable PyTorch backend for GPU acceleration (if available)")
        print("  -b, --bench: run a CPU (NumPy) vs GPU (Torch) speed benchmark")
        print("  --repeats=N: number of runs to average in benchmark (default 3)")
        print("  --device=...: pick cuda|mps|cpu (default avoids MPS; uses CUDA if available, else CPU)")
        print("  --dtype=...: set Torch dtype (float32/float64; default float32 on GPU)")
        print("  --compile: try torch.compile (CUDA) for kernel fusion")
        sys.exit(1)

    if run_bench:
        benchmark(parfile, device_str=device_str, repeats=bench_repeats, do_write=False)
        sys.exit(0)

    # Run the main function with the parameter file and visualization flag
    main(parfile, visualize=visualize, use_torch=use_torch, device_str=device_str, cli_dtype=_cli_dtype, cli_compile=_cli_compile)
