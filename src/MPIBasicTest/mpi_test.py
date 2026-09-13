from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
node = MPI.Get_processor_name()

print(f"Hello from rank {rank} of {size} on node {node}")