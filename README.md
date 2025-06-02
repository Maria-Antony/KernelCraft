# 🧠 KernelCraft Lite

**KernelCraft Lite** is an interactive Triton-based GPU kernel visualizer and profiler for AI/LLM inference. It showcases essential operations used in Transformers—like Softmax, LayerNorm, and GELU—executed on GPU via custom Triton kernels.

Built using:
- [Triton](https://github.com/openai/triton) for GPU kernel programming
- [Streamlit](https://streamlit.io/) for visual interface
- [PyTorch](https://pytorch.org/) for tensor handling

---

## 🎯 Features

- 🔧 Run and benchmark multiple Triton GPU kernels
- 📊 Real-time execution profiling with bar chart visualizations
- 🤖 Kernels relevant to LLMs: Softmax, LayerNorm, Dropout, GELU + Bias
- 🧮 Core tensor ops: MatMul, Add, ReLU

---

## 🚀 Demo

> A live visual UI to understand how low-level GPU kernels behave under inference loads.


---

## 🧱 Included Kernels

| Kernel           | Purpose                            |
|------------------|-------------------------------------|
| `MatMul`         | Core matrix multiplication          |
| `Softmax`        | Attention computation in LLMs       |
| `Elementwise Add`| Simple tensor addition              |
| `ReLU Activation`| Feedforward network activation      |
| `LayerNorm`      | Transformer normalization           |
| `Dropout`        | Regularization (training)           |
| `Bias + GELU`    | Transformer FFN activation          |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/kernelcraft-lite.git
cd kernelcraft-lite
pip install -r requirements.txt
