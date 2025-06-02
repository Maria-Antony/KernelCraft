import triton
import triton.language as tl
import torch

@triton.jit
def gelu_bias_kernel(X_ptr, B_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask)
    b = tl.load(B_ptr + offs, mask=mask)
    z = x + b

    y = 0.5 * z * (1.0 + tl.tanh(0.797885 * (z + 0.044715 * z * z * z)))
    tl.store(Y_ptr + offs, y, mask=mask)

def run_gelu_bias():
    N = 1024
    x = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    y = torch.empty_like(x)

    gelu_bias_kernel[(N // 1024,)](x, b, y, N, BLOCK_SIZE=1024)
    return x, b, y
