# cnn-fashion-mnist-comparison/README.md

# CNN on FashionMNIST (Training, Validation, Testing)

This project demonstrates the implementation of various Convolutional Neural Network (CNN) architectures on the FashionMNIST dataset. The goal is to load the dataset, train different models, and compare their performance metrics.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [License](#license)

## Overview

The FashionMNIST dataset consists of 70,000 grayscale images of clothing items, divided into 10 categories. This project includes the following CNN architectures:

1. Multi-Layer Perceptron (MLP)
2. TinyVGG
3. AlexNet
4. VGG16

The notebook provides a comprehensive analysis of each model's performance, including training time, accuracy, and loss metrics.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/cnn-fashion-mnist-comparison.git
cd cnn-fashion-mnist-comparison
pip install -r requirements.txt
```

## Usage

To run the Jupyter notebook and execute the code, use the following command:

```bash
jupyter notebook notebooks/cnn_fashion_mnist_comparison.ipynb
```

Follow the instructions in the notebook to load the dataset, train the models, and visualize the results.

## Models

The following models are implemented in the `src/models` directory:

- **MLP**: A simple feedforward neural network.
- **TinyVGG**: A compact version of the VGG architecture.
- **AlexNet**: A classic CNN architecture known for its performance on image classification tasks.
- **VGG16**: A deeper version of the VGG architecture with 16 layers.

## Results

The results of the training process, including accuracy and loss metrics for each model, are displayed in the notebook. Visualizations of the training and validation metrics are also provided.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.