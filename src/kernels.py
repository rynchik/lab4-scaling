"""Матрично-векторные операции ЛР1–ЛР2 с согласованной областью замера."""

from mpi4py import MPI
import numpy as np

from .transport import partition, scatter_matrix, scatter_vector


def matvec(A, x, M, N, comm, method="scatterv", cached=None):
    local_A = cached if cached is not None else scatter_matrix(A, M, N, comm, method)
    x = x.copy() if comm.rank == 0 else np.empty(N)
    comm.Bcast(x, root=0)
    local_b = local_A @ x
    counts, offsets = partition(M, comm.size)
    answer = np.empty(M) if comm.rank == 0 else None
    if method == "sendrecv" and cached is None:
        if comm.rank == 0:
            answer[: counts[0]] = local_b
            for rank in range(1, comm.size):
                first = int(offsets[rank])
                last = first + int(counts[rank])
                comm.Recv(answer[first:last], source=rank, tag=20)
        else:
            comm.Send(local_b, dest=0, tag=20)
    else:
        comm.Gatherv(
            local_b,
            [answer, counts, offsets, MPI.DOUBLE] if comm.rank == 0 else None,
            root=0,
        )
    return answer


def dot(b, M, comm, all_ranks=False):
    local = scatter_vector(b, M, comm)
    partial = np.array([local @ local])
    answer = np.empty(1)
    if all_ranks:
        comm.Allreduce(partial, answer, op=MPI.SUM)
    else:
        comm.Reduce(partial, answer if comm.rank == 0 else None, op=MPI.SUM, root=0)
    return float(answer[0]) if all_ranks or comm.rank == 0 else None


def transpose(A, b, M, N, comm):
    local_A = scatter_matrix(A, M, N, comm)
    local_b = scatter_vector(b, M, comm)
    partial = local_A.T @ local_b
    answer = np.empty(N) if comm.rank == 0 else None
    comm.Reduce(partial, answer, op=MPI.SUM, root=0)
    return answer


def repeated_matvec(A, x, M, N, comm, reuse=False, repeats=10):
    """Полная цена серии, включая разовую раздачу при reuse=True."""
    cached = scatter_matrix(A, M, N, comm) if reuse else None
    answer = None
    for _ in range(repeats):
        answer = matvec(A, x, M, N, comm, cached=cached)
    return answer
