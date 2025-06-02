import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N

    a_offset = row * K + tl.arange(0, BLOCK_SIZE)
    b_offset = tl.arange(0, BLOCK_SIZE) * N + col

    a = tl.load(A_ptr + a_offset)
    b = tl.load(B_ptr + b_offset)
    acc = tl.sum(a * b, axis=0)

    out_offset = row * N + col
    tl.store(C_ptr + out_offset, acc)

def run_matmul():
    M, K, N = 16, 64, 16
    A = torch.randn((M, K), device='cuda')
    B = torch.randn((K, N), device='cuda')
    C = torch.empty((M, N), device='cuda')

    matmul_kernel[(M*N,)](A, B, C, M, N, K, BLOCK_SIZE=K)
    return A, B, C
