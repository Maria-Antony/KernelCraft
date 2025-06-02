import triton
import triton.language as tl
import torch

@triton.jit
def dropout_kernel(X_ptr, Y_ptr, mask_ptr, prob, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask)
    rand = tl.rand(offs.to(tl.float32))  # Simple pseudo-random
    keep = rand > prob
    y = x * keep / (1.0 - prob)
    tl.store(Y_ptr + offs, y, mask=mask)
    tl.store(mask_ptr + offs, keep, mask=mask)

def run_dropout():
    N = 1024
    x = torch.randn(N, device='cuda')
    y = torch.empty_like(x)
    mask = torch.empty_like(x)
    prob = 0.1

    dropout_kernel[(N // 1024,)](x, y, mask, prob, N, BLOCK_SIZE=1024)
    return x, y, mask
