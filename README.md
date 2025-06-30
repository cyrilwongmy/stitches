# Stitches: The Needle Deep Learning Library

A complete implementation of the Needle deep learning library from [the dlsys course](https://dlsyscourse.org/). This repository contains a fully functional deep learning framework designed for educational purposes, featuring automatic differentiation, tensor operations, and neural network training capabilities.

## 🚀 Features

- **Automatic Differentiation**: Complete autograd system with backward pass computation
- **Tensor Operations**: Comprehensive set of mathematical operations (matmul, relu, log, exp, etc.)
- **Neural Network Training**: Ready-to-use training utilities with SGD optimization
- **MNIST Support**: Built-in MNIST dataset parsing and training examples
- **NumPy Backend**: Efficient computation using NumPy as the underlying backend
- **Easy Debugging**: Debugging is easy with the test suite in command line or vscode debugger.

## 📁 Project Structure

```
stitches/
├── python/needle/          # Core Needle library
│   ├── autograd.py        # Automatic differentiation engine
│   ├── backend_numpy.py   # NumPy backend implementation
│   ├── ops/              # Mathematical operations
│   └── init/             # Tensor initialization utilities
├── apps/                  # Example applications
│   └── simple_ml.py      # MNIST neural network training
├── data/                  # MNIST dataset files
├── tests/                 # Test suite
└── hw_notebook/          # Jupyter notebooks for homework (for reference only)
```

## 🛠️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cyrilwongmy/stitches.git
   cd stitches
   ```

2. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

## 🧪 Testing

Run the test suite to verify the implementation:

```bash
# Test the autograd module's compute_gradient function
python3 -m pytest -k "compute_gradient"

# Run all tests
python3 -m pytest
```

## 📚 Example Usage

```python
import sys
sys.path.append("python/")
import needle as ndl

# Create tensors
x = ndl.Tensor([[1, 2], [3, 4]])
y = ndl.Tensor([[5, 6], [7, 8]])

# Perform operations
z = ndl.matmul(x, y)
result = ndl.relu(z)

# Compute gradients
result.backward()
```

## 🏗️ Development

This repository represents the **final state** of the Needle library implementation. The main branch contains the complete, working implementation that can be used for:

- Learning deep learning fundamentals
- Understanding automatic differentiation
- Building and training neural networks
- Educational purposes and research

## 📖 Learning Resources

- [dlsys course](https://dlsyscourse.org/) - Original course materials
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Handwritten digit recognition dataset

## 🐛 Report Bugs

If you find any bugs or have any suggestions, please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

---

**Note**: This is the main branch containing the complete implementation (incomplete). For homework-specific versions or incomplete implementations, check out the corresponding homework branches (hw1, hw2, hw3, hw4).
