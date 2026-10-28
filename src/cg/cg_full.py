"""Полная версия: x, r, p, q распределены по компонентам между рангами."""

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
    local_N = int(counts[comm.rank])
    # Нулевое начальное приближение создаётся локально: рассылать нули не требуется.
    x = np.zeros(local_N, dtype=np.float64)
    full = np.empty(N, dtype=np.float64)
    comm.Allgatherv(x, [full, counts, offsets, MPI.DOUBLE])
    data_residual = local_b - local_A @ full
    r = np.empty(local_N, dtype=np.float64)
    comm.Reduce_scatter(local_A.T @ data_residual, r, recvcounts=counts, op=MPI.SUM)
    p = r.copy()
    rr = comm.allreduce(float(r @ r), op=MPI.SUM)
    initial_norm = np.sqrt(rr)
    history = [float(initial_norm)]
    threshold = max(atol, tol * initial_norm)
    reason = "tolerance" if initial_norm <= threshold else "maxiter"
    q = np.empty(local_N, dtype=np.float64)
    comm.Barrier()
    start = MPI.Wtime()
    if initial_norm > threshold:
        for _ in range(maxiter):
            comm.Allgatherv(p, [full, counts, offsets, MPI.DOUBLE])
            u = local_A @ full
            comm.Reduce_scatter(local_A.T @ u, q, recvcounts=counts, op=MPI.SUM)
            q += alpha * p
            denominator = comm.allreduce(float(p @ q), op=MPI.SUM)
            if not np.isfinite(denominator) or denominator <= 0:
                reason = "nonpositive_curvature"
                break
            step = rr / denominator
            x += step * p
            r -= step * q
            rr_new = comm.allreduce(float(r @ r), op=MPI.SUM)
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
    return finish(comm, x, counts, offsets, False, history, elapsed, reason)


if __name__ == "__main__":
    run_cli(solve, "full")
