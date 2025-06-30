# Stitches: The Needle Pieces together

This repository contains the Needle library homeworks together from [the dlsys course](https://dlsyscourse.org/), a deep learning library designed for educational purposes. This repository is intended to help learners debug and finish the homework assignments in an easy way.

## Homeworks

Each homework without a solution is put into a separate branch and can be accessed by checking out the corresponding branch. The branches are named as follows:
- `hw1`: Homework 1
- `hw2`: Homework 2
- `hw3`: Homework 3
- `hw4`: Homework 4

## Homework 1: Autograd and Basic Operations

This branch focuses on implementing the core components of the Needle deep learning library:

### Key Components

1. **Automatic Differentiation (`autograd.py`)**
   - Tensor class with automatic gradient computation
   - Computational graph construction and traversal
   - Backward pass implementation

2. **Mathematical Operations (`ops/`)**
   - Basic arithmetic operations (add, subtract, multiply, divide)
   - Matrix operations (matmul, transpose)
   - Element-wise operations (power, log, exp)
   - Reduction operations (summation, broadcast)

3. **Simple ML Application (`apps/simple_ml.py`)**
   - MNIST data loading and preprocessing
   - Softmax loss function
   - Two-layer neural network training with SGD

### Getting Started

1. **Setup the environment:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Run tests:**
   ```bash
   # Test the autograd module's compute_gradient function
   python3 -m pytest -k "compute_gradient"
   
   # Run all autograd tests
   python3 -m pytest tests/test_autograd_hw.py
   
   # Test the simple ML application
   python3 -m pytest tests/test_simple_ml.py
   ```

3. **Work with the notebook:**
   ```bash
   # Open the homework notebook
   jupyter notebook hw_notebook/hw1.ipynb
   ```

### Dataset

The project includes the MNIST dataset in the `data/` directory:
- `train-images-idx3-ubyte.gz` - Training images
- `train-labels-idx1-ubyte.gz` - Training labels  
- `t10k-images-idx3-ubyte.gz` - Test images
- `t10k-labels-idx1-ubyte.gz` - Test labels

### Implementation Status

The following components need to be implemented:

- **`parse_mnist()`** in `apps/simple_ml.py` - MNIST data loading
- **`softmax_loss()`** in `apps/simple_ml.py` - Softmax loss function
- **`nn_epoch()`** in `apps/simple_ml.py` - Neural network training
- Various mathematical operations in the `ops/` directory
- Autograd functionality in `autograd.py`

## Other Homework Branches

Each homework is organized in separate branches:
- `hw1`: Homework 1 - Autograd and Basic Operations (current)
- `hw2`: Homework 2 - Coming soon
- `hw3`: Homework 3 - Coming soon  
- `hw4`: Homework 4 - Coming soon

## Contributing

This is an educational repository. Feel free to:
- Report bugs or issues
- Suggest improvements
- Share your solutions (in separate branches)

## License

This project is part of the dlsys course materials. Please refer to the original course repository for licensing information.