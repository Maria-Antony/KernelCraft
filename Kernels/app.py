import streamlit as st
import matplotlib.pyplot as plt

from profiler import profile_kernel
from matmul import run_matmul
from softmax import run_softmax
from add import run_add
from relu import run_relu
from layernorm import run_layernorm
from dropout import run_dropout
from gelu_bias import run_gelu_bias

# Title and Description
st.set_page_config(page_title="KernelCraft Lite", layout="centered")
st.title("🧠 KernelCraft Lite")
st.caption("Visualize and benchmark Triton GPU kernels used in LLMs and deep learning.")

# Kernel mapping
kernels = {
    "MatMul": run_matmul,
    "Softmax": run_softmax,
    "Elementwise Add": run_add,
    "ReLU Activation": run_relu,
    "LayerNorm": run_layernorm,
    "Dropout": run_dropout,
    "Bias + GELU": run_gelu_bias
}

# User selection
selected_kernel = st.selectbox("📌 Choose a kernel to run:", list(kernels.keys()))
run_button = st.button("🚀 Run Kernel")

# Storage for execution times
if "history" not in st.session_state:
    st.session_state.history = {}

# Execution and profiling
if run_button:
    kernel_fn = kernels[selected_kernel]
    exec_time = profile_kernel(kernel_fn)
    st.success(f"✅ {selected_kernel} executed in {exec_time * 1000:.2f} ms")
    
    st.session_state.history[selected_kernel] = exec_time * 1000

    # Display bar chart of all recorded runs
    st.subheader("📊 Execution Time Comparison")
    fig, ax = plt.subplots()
    labels = list(st.session_state.history.keys())
    times = [st.session_state.history[k] for k in labels]
    ax.bar(labels, times)
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Kernel")
    ax.set_title("Triton Kernel Execution Time")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Triton, PyTorch, and Streamlit.")
