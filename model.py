"""
Full definition of a 3D-Parallel GPT Language Model, all of it in this single file.
For ease of presentation: 
- we ignore dropout, special initialize/weight decay techniques, mixed precision, and so on. 
- we focus on 3D parallelism: GPipe, tensor parallelism and data parallelism with All-reduce. 

References:
1) karpathy/minGPT:
https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
2) NVIDIA/Megatron-LM:
https://github.com/NVIDIA/Megatron-LM
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import utils

# ----------Tensor Parallelism Transformer Block----------

class ParallelAttention(nn.Module):
    """Multi-head causal self-attention layer with Tensor Parallelism. """
    def __init__(self, config):
        super().__init__()
        self.n_local_head = config.n_head // config.model_parallel_size
        self.n_local_embd = config.n_embd // config.model_parallel_size
        self.head_dim = config.n_embd // config.n_head

        # column-parallel linear layer for QKV projections
        self.c_attn = nn.Linear(config.n_embd, 3 * self.n_local_embd, bias=False)
        # row-parallel linear layer for output projection 
        self.c_proj = nn.Linear(self.n_local_embd, config.n_embd, bias=False) 
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, _ = x.size() # batch size, sequence length

        # calculate query, key, values
        q, k ,v  = self.c_attn(x).split(self.n_local_embd, dim=2)
        k = k.view(B, T, self.n_local_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_local_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_local_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_local_embd)

        # output projection, followed by an all-reduce layer
        y = self.c_proj(y)
        if tp_size > 1:
            y_sum = torch.distributed.nn.all_reduce(y, group=todo) # with backward's allreduce
        return y_sum

class ParallelMLP(nn.Module):
    """MLP layer with Tensor Parallelism. """
    def __init__(self, config):
        super().__init__()
        self.n_local_embd = config.n_embd // config.model_parallel_size
        self.c_fc = nn.Linear(config.n_embd, 4 * self.n_local_embd, bias=False) # column-parallel linear layer
        self.c_proj = nn.Linear(4 * self.n_local_embd, config.n_embd, bias=False) # row-parallel linear layer
        self.act = torch.nn.GELU(approximate='tanh') # GeLU activation function

    def forward(self, x):
        y = self.c_proj(self.act(self.c_fc(x)))
        if tp_size > 1:
            y_sum = torch.distributed.nn.all_reduce(y, group=todo) # with backward's allreduce
        return y_sum

class TransformerBlock(nn.Module):
    """Transformer block with Tensor Parallelism. """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = ParallelAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = ParallelMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

# ----------Pipeline Parallelism Stage (GPipe)----------

class PipelineStage(nn.module):
    """Pipeline parallelism stage with a subset of Transformer blocks"""
    def __init__(self, config):
        super().__init__()
        self.pipeline_size = config.pipeline_size
        self.pipeline_rank = todo
        self.is_first_pipeline_stage = (self.pipeline_rank == 0)
        self.is_last_pipeline_stage = (self.pipeline_rank == (self.pipeline_size-1))
        self.n_local_layer = config.n_layer // self.pipeline_size
        self.n_embd = config.n_embd

        # input, output, and transformer layer
        if self.is_first_pipeline_stage:
            self.input_layers = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd)
            ))
            # broadcast embedding weights (TP), todo
        if self.is_last_pipeline_stage:
            self.output_layers = nn.ModuleDict(dict(
                post_ln = nn.LayerNorm(config.n_embd), 
                lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            ))
            # broadcast output head weight (TP), todo
        self.transformer_layers = torch.nn.ModuleList([TransformerBlock(config) for _ in range(self.n_local_layer)])

    def forward_step(self, idx, targets):
        """forward pass: idx or intermediate input-> loss or intermediate output"""
        if self.is_first_pipeline_stage: # input layers
            pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device).unsqueeze(0)
            tok_emb = self.input_layers.wte(idx)
            pos_emb = self.input_layers.wpe(pos)
            x = tok_emb + pos_emb
            input_tensor = None
        else:
            tensor_shape = (idx.size(0), idx.size(1), self.n_embd)
            x = utils.forward_recv(tensor_shape) # receive input from previous stage
            input_tensor = x # used for input grad tensor

        for block in self.transformer_layers: # transformer layers
            x = block(x)
        
        if self.is_last_pipeline_stage: # output layers
            logits = self.output_layers.lm_head(self.output_layers.post_ln(x))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return input_tensor, loss
        else:
            utils.forward_send(x) # send output to next stage
            return input_tensor, x

    def backward_step(self, input_tensor, output_tensor):
        """backward pass: output grad -> input grad."""
        if input_tensor is not None:
            input_tensor.retain_grad()  # retain input grad

        if self.is_last_pipeline_stage:
            output_tensor_grad = None  # output is loss scalar
        else:
            tensor_shape = (output_tensor.size(0), output_tensor.size(1), self.n_embd)
            output_tensor_grad = utils.backward_recv(tensor_shape) # receive output grad from next stage

        torch.autograd.backward(tensors=output_tensor, grad_tensors=output_tensor_grad) # backward

        if input_tensor is not None:
            input_grad = input_tensor.grad
            utils.backward_send(input_grad)


# ----------3D Parallelism GPT----------

class GPT3D(nn.Module):
    """GPT Language Model with 3D Parallelism."""
    def __init__(self, config):
        super().__init__()
        self.model = self.PipelineStage(config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), config.learning_rate, weight_decay=config.weight_decay)
        self.n_microbatch = config.n_microbatch

    def pipeline_step(self, idx, targets):
        """GPipe: all forward and all backward passes."""
        input_tensors = []
        output_tensors = []

        idx_chunks = idx.chunk(self.n_microbatch)
        target_chunks = targets.chunk(self.n_microbatch)

        for i in range(self.n_microbatch):
            input_tensor, output_tensor = self.model.forward_step(idx_chunks[i], target_chunks[i])
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
        
        for i in range(self.n_microbatch):
            self.model.backward_step(input_tensors[i], output_tensors[i])

    def non_pipeline_step(self, idx, targets):
        """1F1B gradient accumulation without pipeline."""
        idx_chunks = idx.chunk(self.n_microbatch)
        target_chunks = targets.chunk(self.n_microbatch)

        for i in range(self.n_microbatch):
            input_tensor, output_tensor = self.model.forward_step(idx_chunks[i], target_chunks[i])
            self.model.backward_step(input_tensor, output_tensor)

    def optimizer_step(self):
        """Data Parallelism: All-reduce grads before parameter update."""
        pass
