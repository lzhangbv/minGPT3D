import torch
import torch.distributed as dist


# init


# data parallelism helper



# tensor parallelism helper
def tensor_parallel_all_reduce(input_):
    """All-reduce the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
    return input_

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return tensor_parallel_all_reduce(grad_output)

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        return tensor_parallel_all_reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return tensor_parallel_all_reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def copy_to_tensor_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)

def reduce_from_tensor_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


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