import argparse
import logging
import pathlib
import time

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation script for 2D Rayleigh-Benard convection using Dedalus"
    )
    parser.add_argument(
        "--lx",
        default=5.0,
        type=float,
        help="Physical length in x dimension. (default: 5.0)",
    )
    parser.add_argument(
        "--lz",
        default=1.0,
        type=float,
        help="Physical length in z dimension. (default: 1.0)",
    )
    parser.add_argument(
        "--nx",
        default=128,
        type=int,
        help="Simulation resolution in x dimension. (default: 128)",
    )
    parser.add_argument(
        "--nz",
        default=64,
        type=int,
        help="Simulation resolution in z dimension. (default: 64)",
    )
    parser.add_argument(
        "--dt",
        default=1e-5,
        type=float,
        help="Simulation time-step size in seconds. (default: 1e-5)",
    )
    parser.add_argument(
        "--stop_sim_time",
        default=50.0,
        type=float,
        help="Simulation stop time in seconds. (default: 50)",
    )
    parser.add_argument(
        "--rayleigh",
        default=15e4,
        type=float,
        help="Simulation Rayleigh number. (default: 15e4)",
    )
    parser.add_argument(
        "--prandtl",
        default=5,
        type=float,
        help="Simulation Prandtl number. (default: 5)",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for initial perturbations. (default: 42)",
    )
    parser.add_argument(
        "--logging_period",
        default=10,
        type=int,
        help="Each [logging_period] iterations, a log with the solver statistics will be displayed. (default: 10)",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()
    # Aspect ration and resolution
    Lx, Lz = (args.lx, args.lz)
    nx, nz = (args.nx, args.nz)
    # Physical constants
    Prandtl = args.prandtl
    Rayleigh = args.rayleigh

    fname = f"snap_Pr_{Prandtl}_Ra_{int(Rayleigh)}"  # file name where experiments will be stored

    # Create bases and domain
    x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3 / 2)
    z_basis = de.Chebyshev("z", nz, interval=(0, Lz), dealias=3 / 2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

    problem = de.IVP(domain, variables=["p", "u", "w", "uz", "wz", "T", "Tz"])
    problem.meta["p", "T", "u", "w"]["z"]["dirichlet"] = True
    problem.parameters["Pr"] = Prandtl
    problem.parameters["Ra"] = Rayleigh
    # Heat Transfer Equation
    problem.add_equation("dt(T) - (1 / Pr)*(dx(dx(T)) + dz(Tz)) = -(u*dx(T) + w*Tz)")
    # Momentum equations in x-z
    problem.add_equation("dt(u) + dx(p) - dx(dx(u)) - dz(uz) = -(u*dx(u) + w*uz)")
    problem.add_equation(
        "dt(w) + dz(p) - dx(dx(w)) - dz(wz) - (Ra / Pr)*T = -(u*dx(w) + w*wz)"
    )
    # Mass Conservation Equation
    problem.add_equation("dx(u) + wz = 0")
    # u_x - du/dx = 0
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_equation("Tz - dz(T) = 0")

    # Fixed temperature at top boundary
    problem.add_bc("right(T) = 0")
    # Fixed flux on bottom boundary
    problem.add_bc("left(Tz) = -1")
    # No slip boundary conditions
    problem.add_bc("left(u) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(u) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("right(p) = 0", condition="(nx == 0)")

    solver = problem.build_solver(de.timesteppers.RK443)

    x = domain.grid(0)
    z = domain.grid(1)
    T = solver.state["T"]
    Tz = solver.state["Tz"]
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=args.seed)
    noise = rand.standard_normal(gshape)[slices]
    # Linear background + perturbations damped at walls
    zb, zt = z_basis.interval
    pert = 1e-1 * noise * (zt - z) * (z - zb)
    T["g"] += pert
    T.differentiate("z", out=Tz)

    solver.stop_sim_time = args.stop_sim_time
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    dt = args.dt

    # Analysis files
    snapshots = solver.evaluator.add_file_handler(fname, sim_dt=1e-3, max_writes=200)
    snapshots.add_task(
        "0.5 * (u ** 2 + w ** 2)", layout="g", name="KE"
    )  # Kinetic energy
    snapshots.add_task(
        "sqrt(u ** 2 + w ** 2)", layout="g", name="|uvec|"
    )  # Norm of the velocity
    snapshots.add_system(solver.state)

    # CFL
    CFL = flow_tools.CFL(
        solver,
        initial_dt=dt,
        cadence=10,
        safety=0.5,
        max_change=1.5,
        min_change=1,
        max_dt=1e-3,
        threshold=0.05,
    )
    CFL.add_velocities(("u", "w"))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
    flow.add_property("sqrt(u*u + w*w) / Ra", name="Re")

    # Start solving
    logger.info("Starting loop")
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        solver.step(dt)
        if solver.iteration % args.logging_period == 0:
            logger.info(
                "Iteration: %i, Time: %e, dt: %e"
                % (solver.iteration, solver.sim_time, dt)
            )

    end_time = time.time()

    # Print statistics
    logger.info("Run time: %f" % (end_time - start_time))
    logger.info("Iterations: %i" % solver.iteration)

    # Merge the snapshot files together
    post.merge_process_files(fname, cleanup=True)
    set_paths = list(pathlib.Path(fname).glob(f"{fname}_s*.h5"))
    post.merge_sets(f"{fname}/snapshots.h5", set_paths, cleanup=True)
