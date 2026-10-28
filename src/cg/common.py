"""Общие операции ввода, разбиения и завершения MPI-решателей."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI
import numpy as np


@dataclass
class Result:
    x: object
    iterations: int
    converged: bool
    history: list
    kernel_seconds: float
    stop_reason: str


def partition(length, size):
    if length < 0 or size <= 0:
        raise ValueError("Требуются length >= 0 и size > 0")
    base, extra = divmod(length, size)
    counts = np.full(size, base, dtype=np.int64)
    counts[:extra] += 1
    offsets = np.zeros(size, dtype=np.int64)
    offsets[1:] = np.cumsum(counts[:-1])
    return counts, offsets


def on_root(comm, load):
    data = error = None
    if comm.rank == 0:
        try:
            data = load()
        except (OSError, ValueError, TypeError) as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error is not None:
        raise ValueError(error)
    return data


def validate(A, b, alpha, tol, atol, maxiter):
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if A.ndim != 2 or min(A.shape) <= 0 or b.shape != (A.shape[0],):
        raise ValueError("Требуется A размера M x N и b длины M")
    if not np.isfinite(b).all() or any(
        not np.isfinite(A[start:][:1024]).all() for start in range(0, A.shape[0], 1024)
    ):
        raise ValueError("Все элементы должны быть конечными числами")
    if not all(np.isfinite(v) and v >= 0 for v in (alpha, tol, atol)):
        raise ValueError("alpha, tol, atol должны быть конечными и неотрицательными")
    if maxiter is None:
        maxiter = 4 * A.shape[1]
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError("maxiter должен быть положительным целым")
    return A, b, float(alpha), float(tol), float(atol), int(maxiter)


def distribute(A, b, alpha, tol, atol, maxiter, comm):
    """Раздача данных по окнам, безопасная для исходных размеров ЛР4."""
    from ..transport import scatter_matrix, scatter_vector

    data = on_root(comm, lambda: validate(A, b, alpha, tol, atol, maxiter))
    if comm.rank == 0:
        A, b, alpha, tol, atol, maxiter = data
    dims = np.array(A.shape if comm.rank == 0 else [0, 0], dtype=np.int64)
    comm.Bcast(dims, root=0)
    alpha, tol, atol, maxiter = comm.bcast(
        (alpha, tol, atol, maxiter) if comm.rank == 0 else None, root=0
    )
    M, N = map(int, dims)
    cols, col_offsets = partition(N, comm.size)
    local_A = scatter_matrix(A, M, N, comm)
    local_b = scatter_vector(b, M, comm)
    return local_A, local_b, N, cols, col_offsets, alpha, tol, atol, maxiter


def read_data(directory):
    directory = Path(directory)
    params = (directory / "in.dat").read_text(encoding="utf-8").split()
    if len(params) != 2:
        raise ValueError("in.dat должен содержать N, затем M")
    N, M = map(int, params)
    if min(N, M) <= 0:
        raise ValueError("Размеры должны быть положительными")
    A = np.loadtxt(directory / "AData.dat", dtype=np.float64).reshape(-1)
    b = np.loadtxt(directory / "bData.dat", dtype=np.float64).reshape(-1)
    if A.size != M * N:
        raise ValueError("Количество элементов A не равно M * N")
    return A.reshape(M, N), b


def reference_solution(A, b, alpha=0.0):
    """При alpha>0 решается эквивалентная расширенная задача МНК."""
    if alpha == 0:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    augmented_A = np.vstack([A, np.sqrt(alpha) * np.eye(A.shape[1])])
    augmented_b = np.r_[b, np.zeros(A.shape[1])]
    return np.linalg.lstsq(augmented_A, augmented_b, rcond=None)[0]


def metrics(A, b, x, reference, alpha=0.0):
    residual = A @ x - b
    gradient = A.T @ residual + alpha * x
    initial_gradient = A.T @ b
    tiny = np.finfo(float).tiny
    return {
        "relative_solution_error": float(
            np.linalg.norm(x - reference) / max(np.linalg.norm(reference), tiny)
        ),
        "relative_data_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(b), tiny)
        ),
        "relative_normal_residual": float(
            np.linalg.norm(gradient) / max(np.linalg.norm(initial_gradient), tiny)
        ),
    }


def finish(comm, x, counts, offsets, replicated, history, elapsed, reason):
    seconds = comm.reduce(elapsed, op=MPI.MAX, root=0)
    answer = None
    if replicated:
        answer = x.copy() if comm.rank == 0 else None
    else:
        answer = (
            np.empty(int(counts.sum()), dtype=np.float64) if comm.rank == 0 else None
        )
        comm.Gatherv(
            x, [answer, counts, offsets, MPI.DOUBLE] if comm.rank == 0 else None, root=0
        )
    return Result(
        answer,
        len(history) - 1,
        reason == "tolerance",
        history,
        float(seconds) if comm.rank == 0 else 0.0,
        reason,
    )


def run_cli(solve, version):
    parser = argparse.ArgumentParser(description=f"CG для МНК: {version}")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--maxiter", type=int, default=None)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    try:
        data = on_root(comm, lambda: read_data(args.data_dir))
        A, b = data if comm.rank == 0 else (None, None)
        result = solve(
            A,
            b,
            alpha=args.alpha,
            tol=args.tol,
            atol=args.atol,
            maxiter=args.maxiter,
            comm=comm,
        )
    except ValueError as exc:
        raise SystemExit(f"Input error: {exc}") from exc
    if comm.rank == 0:
        import json

        reference = reference_solution(A, b, args.alpha)
        error = metrics(A, b, result.x, reference, args.alpha)
        np.savetxt(f"Results_{version}.dat", result.x, fmt="%.16e")
        print(
            json.dumps(
                {
                    "version": version,
                    "processes": comm.size,
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "stop_reason": result.stop_reason,
                    "kernel_seconds": result.kernel_seconds,
                    **error,
                },
                indent=2,
            )
        )
    if not result.converged:
        raise SystemExit("CG did not converge: " + result.stop_reason)
