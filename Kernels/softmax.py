import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(X_ptr, Y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offsets = row_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    softmax = num / denom
    tl.store(Y_ptr + offsets, softmax, mask=mask)

def run_softmax():
    BLOCK_SIZE = 1024
    x = torch.randn(BLOCK_SIZE, device='cuda')
    y = torch.empty_like(x)

    softmax_kernel[(1,)](x, y, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return x, y
