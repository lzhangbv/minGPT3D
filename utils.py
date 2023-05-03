import torch
import torch.distributed as dist


# init


# data parallelism helper



# tensor parallelism helper



# pipeline parallelism helper
def forward_send(tensor):
    pass

def forward_recv(tensor_shape):
    pass

def backward_send(tensor):
    pass

def backward_recv(tensor_shape):
    pass

def send(tensor, dest_rank):
    return dist.send(tensor, dest_rank)

def recv(tensor, src_rank):
    return dist.recv(tensor, src_rank)