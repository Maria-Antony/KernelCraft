import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0)
    tl.store(Y_ptr + offsets, y, mask=mask)

def run_relu():
    N = 1024
    x = torch.randn(N, device='cuda')
    y = torch.empty_like(x)

    relu_kernel[(N // 1024,)](x, y, N, BLOCK_SIZE=1024)
    return x, y
