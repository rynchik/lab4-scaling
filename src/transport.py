"""Блочная раздача матриц, включая данные с более чем 2**31 элементами."""

from mpi4py import MPI
import numpy as np


def partition(length, size):
    if length < 0 or size <= 0:
        raise ValueError("Некорректные length или size")
    base, extra = divmod(length, size)
    counts = np.full(size, base, dtype=np.int64)
    counts[:extra] += 1
    offsets = np.zeros(size, dtype=np.int64)
    offsets[1:] = np.cumsum(counts[:-1])
    return counts, offsets


def window(counts, offsets, first, last):
    """Количества и смещения относительно текущего окна матрицы."""
    starts = np.maximum(offsets, first)
    stops = np.minimum(offsets + counts, last)
    lengths = np.maximum(stops - starts, 0)
    relative = np.where(lengths > 0, starts - first, 0)
    local = np.where(lengths > 0, starts - offsets, 0)
    return lengths, relative, local


def scatter_matrix(
    A, M, N, comm=MPI.COMM_WORLD, method="scatterv", chunk_bytes=16 * 1024 * 1024
):
    counts, offsets = partition(M, comm.size)
    local = np.empty((int(counts[comm.rank]), N), dtype=np.float64)
    block_rows = max(1, min(chunk_bytes // (8 * N), (2**31 - 1) // N))
    if method == "sendrecv":
        for rank in range(comm.size):
            for start in range(0, int(counts[rank]), block_rows):
                stop = min(start + block_rows, int(counts[rank]))
                if comm.rank == 0:
                    global_start = int(offsets[rank]) + start
                    global_stop = int(offsets[rank]) + stop
                    block = A[global_start:global_stop]
                    if rank == 0:
                        local[start:stop] = block
                    else:
                        comm.Send(block, dest=rank, tag=10)
                elif comm.rank == rank:
                    comm.Recv(local[start:stop], source=0, tag=10)
    elif method == "scatterv":
        for first in range(0, M, block_rows):
            last = min(first + block_rows, M)
            lengths, displacements, local_starts = window(counts, offsets, first, last)
            start = int(local_starts[comm.rank])
            stop = start + int(lengths[comm.rank])
            comm.Scatterv(
                (
                    [A[first:last], lengths * N, displacements * N, MPI.DOUBLE]
                    if comm.rank == 0
                    else None
                ),
                local[start:stop],
                root=0,
            )
    else:
        raise ValueError("Неизвестный способ распределения")
    return local


def scatter_vector(vector, length, comm=MPI.COMM_WORLD):
    counts, offsets = partition(length, comm.size)
    local = np.empty(int(counts[comm.rank]), dtype=np.float64)
    comm.Scatterv(
        [vector, counts, offsets, MPI.DOUBLE] if comm.rank == 0 else None, local, root=0
    )
    return local
