# PyTorch Introduction

A structured project for learning and demonstrating basic PyTorch operations and tensor manipulations.

## Project Structure

```
pytorch_introduction/
├── README.md               # This file
├── requirements.txt        # Project dependencies
├── setup.py               # Package installation configuration
├── main.py                # Main script to demonstrate functionality
├── notebooks/             # Jupyter notebooks for interactive exploration
│   └── pytorch_introduction.ipynb
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── device_utils.py    # GPU/CPU device handling
│   ├── tensor_operations.py  # Basic tensor operations
│   ├── image_processing.py  # Image-related tensor operations
│   └── tensor_statistics.py # Statistical operations on tensors
└── tests/                 # Unit tests
    ├── __init__.py
    ├── test_tensor_operations.py
    ├── test_image_processing.py
    └── test_tensor_statistics.py
```

## Features

This project demonstrates:

1. Basic PyTorch tensor operations
2. GPU acceleration with PyTorch
3. Image representation and manipulation using tensors
4. Statistical analysis of tensor data
5. Best practices for PyTorch project structure

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pytorch_introduction

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

Run the main script to see demonstrations of all functionality:

```bash
python main.py
```

Or explore the notebooks interactively:

```bash
jupyter notebook notebooks/pytorch_introduction.ipynb
```

## Requirements

-   Python 3.7+
-   PyTorch 1.8+
-   CUDA (optional, for GPU acceleration)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
