"""Профилирование блокирующих вызовов CG и оптимизация скалярных редукций."""

from mpi4py import MPI
import numpy as np


class ProfileComm:
    def __init__(self, comm, typed=False):
        self.comm = comm
        self.typed = typed
        self.active = False
        self.times = {}
        self.calls = {}
        self.scalar_send = np.empty(1)
        self.scalar_recv = np.empty(1)

    def __getattr__(self, name):
        return getattr(self.comm, name)

    def timed(self, name, operation):
        start = MPI.Wtime()
        result = operation()
        if self.active:
            self.times[name] = self.times.get(name, 0.0) + MPI.Wtime() - start
            self.calls[name] = self.calls.get(name, 0) + 1
        return result

    def Barrier(self):
        self.comm.Barrier()
        self.times.clear()
        self.calls.clear()
        self.active = True

    def reduce(self, value, op=MPI.SUM, root=0):
        # В скопированном решателе первый reduce после Barrier завершает замер цикла.
        self.active = False
        return self.comm.reduce(value, op=op, root=root)

    def allreduce(self, value, op=MPI.SUM):
        if self.typed:
            self.scalar_send[0] = value
            self.timed(
                "scalar_Allreduce",
                lambda: self.comm.Allreduce(self.scalar_send, self.scalar_recv, op=op),
            )
            return float(self.scalar_recv[0])
        return self.timed("object_allreduce", lambda: self.comm.allreduce(value, op=op))

    def Allgatherv(self, *args, **kwargs):
        return self.timed("Allgatherv", lambda: self.comm.Allgatherv(*args, **kwargs))

    def Reduce_scatter(self, *args, **kwargs):
        return self.timed(
            "Reduce_scatter", lambda: self.comm.Reduce_scatter(*args, **kwargs)
        )

    def Allreduce(self, *args, **kwargs):
        return self.timed(
            "vector_Allreduce", lambda: self.comm.Allreduce(*args, **kwargs)
        )
