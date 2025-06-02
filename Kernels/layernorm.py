import triton
import triton.language as tl
import torch

@triton.jit
def layernorm_kernel(X_ptr, Y_ptr, W_ptr, B_ptr, M, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N

    x = tl.load(X_ptr + offs, mask=mask)
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) ** 2, axis=0) / N
    norm = (x - mean) / tl.sqrt(var + eps)

    w = tl.load(W_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)
    b = tl.load(B_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)
    y = norm * w + b
    tl.store(Y_ptr + offs, y, mask=mask)

def run_layernorm():
    M, N = 1, 1024
    x = torch.randn((M, N), device='cuda')
    w = torch.ones(N, device='cuda')
    b = torch.zeros(N, device='cuda')
    y = torch.empty((M, N), device='cuda')
    eps = 1e-5

    layernorm_kernel[(M,)](x, y, w, b, M, N, eps, BLOCK_SIZE=N)
    return x, y
