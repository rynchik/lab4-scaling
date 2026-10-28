"""Один MPI-запуск: операции ЛР1–ЛР3, профилирование и проверка точности."""

import argparse
import json
from pathlib import Path
import platform
import sys

from mpi4py import MPI
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load  # noqa: E402
from src.cg.cg_full import solve as full  # noqa: E402
from src.cg.cg_simple import solve as simple  # noqa: E402
from src.cg.common import on_root  # noqa: E402
from src.kernels import dot, matvec, transpose, repeated_matvec  # noqa: E402
from src.profiling import ProfileComm  # noqa: E402
from src.transport import partition, scatter_matrix  # noqa: E402


def relative_error(actual, expected):
    actual, expected = np.atleast_1d(actual), np.atleast_1d(expected)
    return float(
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300)
    )


def microbench(comm, repeats=30):
    rows = []
    for N in (200, 500, 1000):
        counts, offsets = partition(N, comm.size)
        local = np.ones(int(counts[comm.rank]))
        vector = np.ones(N)
        receiver = np.empty(N)
        scalar = np.ones(1)
        scalar_recv = np.empty(1)
        operations = {
            "Bcast": lambda: comm.Bcast(vector, root=0),
            "Allgatherv": lambda: comm.Allgatherv(
                local, [receiver, counts, offsets, MPI.DOUBLE]
            ),
            "Allreduce_vector": lambda: comm.Allreduce(vector, receiver, op=MPI.SUM),
            "Reduce_scatter": lambda: comm.Reduce_scatter(
                vector, local, recvcounts=counts, op=MPI.SUM
            ),
            "Allreduce_scalar": lambda: comm.Allreduce(scalar, scalar_recv, op=MPI.SUM),
        }
        for name, operation in operations.items():
            samples = []
            for trial in range(repeats + 3):
                comm.Barrier()
                start = MPI.Wtime()
                operation()
                elapsed = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
                if comm.rank == 0 and trial >= 3:
                    samples.append(elapsed)
            if comm.rank == 0:
                rows.append(
                    {
                        "operation": name,
                        "N": N,
                        "processes": comm.size,
                        "median_s": float(np.median(samples)),
                        "samples": samples,
                    }
                )
    return rows


def experiment(directory, comm, repeats=3, warmup=1, fixed_iters=None, do_micro=False):
    loaded = on_root(comm, lambda: load(directory))
    A = b = truth = gradient = None
    shape = None
    dot_reference = None
    if comm.rank == 0:
        A, b, truth, gradient = loaded
        shape = A.shape
        dot_reference = float(b @ b)
    M, N = comm.bcast(shape, root=0)
    comm.Barrier()
    start = MPI.Wtime()
    cached = scatter_matrix(A, M, N, comm)
    cache_setup = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
    cases = [
        ("matvec_sendrecv", lambda: matvec(A, truth, M, N, comm, "sendrecv"), b),
        ("matvec_scatterv", lambda: matvec(A, truth, M, N, comm), b),
        ("matvec_cached", lambda: matvec(A, truth, M, N, comm, cached=cached), b),
        ("dot_reduce", lambda: dot(b, M, comm), dot_reference),
        ("dot_allreduce", lambda: dot(b, M, comm, True), dot_reference),
        ("transpose", lambda: transpose(A, b, M, N, comm), gradient),
        ("batch10_baseline", lambda: repeated_matvec(A, truth, M, N, comm), b),
        ("batch10_cached", lambda: repeated_matvec(A, truth, M, N, comm, True), b),
    ]
    rows = []
    for name, operation, reference in cases:
        samples, errors = [], []
        for trial in range(warmup + repeats):
            comm.Barrier()
            start = MPI.Wtime()
            result = operation()
            elapsed = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
            if comm.rank == 0:
                error = relative_error(result, reference)
                if error > 1e-10:
                    raise AssertionError(f"{name}: relative error {error}")
                if trial >= warmup:
                    samples.append(elapsed)
                    errors.append(error)
        if comm.rank == 0:
            rows.append(
                {
                    "method": name,
                    "samples": samples,
                    "median_s": float(np.median(samples)),
                    "max_relative_error": max(errors),
                    "scope": (
                        "cached_kernel"
                        if name == "matvec_cached"
                        else (
                            "10_matvec_batch_with_setup"
                            if name.startswith("batch10_")
                            else "distribution_compute_gather"
                        )
                    ),
                }
            )
    cached = None
    for name, solve, typed in [
        ("cg_full", full, False),
        ("cg_full_typed", full, True),
        ("cg_simple", simple, False),
    ]:
        samples, errors, iterations, phase_samples, comm_samples = [], [], [], [], []
        for trial in range(warmup + repeats):
            proxy = ProfileComm(comm, typed)
            result = solve(
                A,
                b,
                tol=0.0 if fixed_iters else 1e-9,
                atol=0.0 if fixed_iters else 1e-12,
                maxiter=fixed_iters or 4 * N,
                comm=proxy,
            )
            if fixed_iters is None and not result.converged:
                raise RuntimeError(f"{name}: {result.stop_reason}")
            if fixed_iters is not None and result.iterations != fixed_iters:
                raise RuntimeError("Число итераций слабого масштабирования изменилось")
            phases = {}
            keys = comm.bcast(sorted(proxy.times) if comm.rank == 0 else None, root=0)
            for key in keys:
                maximum = comm.reduce(proxy.times.get(key, 0.0), op=MPI.MAX, root=0)
                if comm.rank == 0:
                    phases[key] = maximum
            communication = comm.reduce(sum(proxy.times.values()), op=MPI.MAX, root=0)
            if comm.rank == 0:
                error = relative_error(result.x, truth)
                if error > (1e-6 if fixed_iters else 1e-7):
                    raise AssertionError(f"{name}: relative solution error {error}")
                if trial >= warmup:
                    samples.append(result.kernel_seconds)
                    iterations.append(result.iterations)
                    errors.append(error)
                    phase_samples.append(phases)
                    comm_samples.append(communication)
        if comm.rank == 0:
            rows.append(
                {
                    "method": name,
                    "samples": samples,
                    "median_s": float(np.median(samples)),
                    "max_relative_error": max(errors),
                    "iterations": iterations,
                    "scope": "cg_loop",
                    "phase_samples": phase_samples,
                    "comm_total_samples": comm_samples,
                    "calls_per_solve": proxy.calls,
                }
            )
    micro = microbench(comm) if do_micro else []
    if comm.rank == 0:
        return {
            "M": M,
            "N": N,
            "processes": comm.size,
            "repeats": repeats,
            "warmup": warmup,
            "fixed_iters": fixed_iters,
            "cache_setup_s": cache_setup,
            "rows": rows,
            "microbench": micro,
            "mpi": MPI.Get_library_version().strip(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "hostnames": None,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--fixed-iters", type=int)
    parser.add_argument("--microbench", action="store_true")
    args = parser.parse_args()
    if (
        args.repeats <= 0
        or args.warmup < 0
        or (args.fixed_iters is not None and args.fixed_iters <= 0)
    ):
        parser.error("Некорректное число повторов или итераций")
    comm = MPI.COMM_WORLD
    try:
        result = experiment(
            args.data,
            comm,
            args.repeats,
            args.warmup,
            args.fixed_iters,
            args.microbench,
        )
        hostnames = comm.gather(MPI.Get_processor_name(), root=0)
        if comm.rank == 0:
            result["hostnames"] = hostnames
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Saved: {args.output.name}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        comm.Abort(1)


if __name__ == "__main__":
    main()
