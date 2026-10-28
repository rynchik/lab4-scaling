"""Упрощённая версия: полные x, r, p, q хранятся на каждом ранге."""

from mpi4py import MPI
import numpy as np

if __package__:
    from .common import distribute, finish, run_cli
else:
    from common import distribute, finish, run_cli  # type: ignore[no-redef]


def solve(
    A=None, b=None, alpha=0.0, tol=1e-10, atol=1e-12, maxiter=None, comm=MPI.COMM_WORLD
):
    local_A, local_b, N, counts, offsets, alpha, tol, atol, maxiter = distribute(
        A, b, alpha, tol, atol, maxiter, comm
    )
    x = np.zeros(N, dtype=np.float64)
    r = np.empty(N, dtype=np.float64)
    comm.Allreduce(local_A.T @ (local_b - local_A @ x), r, op=MPI.SUM)
    p = r.copy()
    rr = float(r @ r)
    initial_norm = np.sqrt(rr)
    history = [float(initial_norm)]
    threshold = max(atol, tol * initial_norm)
    reason = "tolerance" if initial_norm <= threshold else "maxiter"
    q = np.empty(N, dtype=np.float64)
    comm.Barrier()
    start = MPI.Wtime()
    if initial_norm > threshold:
        for _ in range(maxiter):
            u = local_A @ p
            comm.Allreduce(local_A.T @ u, q, op=MPI.SUM)
            q += alpha * p
            denominator = float(p @ q)
            if not np.isfinite(denominator) or denominator <= 0:
                reason = "nonpositive_curvature"
                break
            step = rr / denominator
            x += step * p
            r -= step * q
            rr_new = float(r @ r)
            if not np.isfinite(rr_new):
                reason = "nonfinite_residual"
                break
            history.append(float(np.sqrt(rr_new)))
            if history[-1] <= threshold:
                reason = "tolerance"
                break
            p *= rr_new / rr
            p += r
            rr = rr_new
    elapsed = MPI.Wtime() - start
    return finish(comm, x, counts, offsets, True, history, elapsed, reason)


if __name__ == "__main__":
    run_cli(solve, "simple")
