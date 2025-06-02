import time
import torch

def profile_kernel(kernel_fn):
    torch.cuda.synchronize()  # Ensure GPU is ready
    start = time.time()
    kernel_fn()               # Run the kernel
    torch.cuda.synchronize()  # Wait for GPU to finish
    end = time.time()
    return end - start        # Elapsed time in seconds
