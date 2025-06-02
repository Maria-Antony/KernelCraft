import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def run_add():
    N = 1024
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c = torch.empty_like(a)

    add_kernel[(N // 1024,)](a, b, c, N, BLOCK_SIZE=1024)
    return a, b, c
