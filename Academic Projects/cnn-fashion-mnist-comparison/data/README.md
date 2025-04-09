# cnn-fashion-mnist-comparison/data/README.md

# FashionMNIST Dataset Documentation

## Overview
The FashionMNIST dataset is a collection of 70,000 grayscale images of clothing items, divided into 10 categories. Each image is 28x28 pixels in size. This dataset serves as a benchmark for various machine learning models, particularly in the field of computer vision.

## Dataset Structure
The dataset is organized into two main parts:
- **Training Set**: 60,000 images used for training models.
- **Test Set**: 10,000 images used for evaluating model performance.

## Categories
The images in the FashionMNIST dataset belong to the following categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Data Format
- **Image Format**: Each image is stored in a grayscale format with pixel values ranging from 0 to 255.
- **Label Format**: Each image is associated with a label indicating its category, represented as an integer from 0 to 9.

## Usage
To load the FashionMNIST dataset in your project, you can use the following code snippet:

```python
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
```

This will download the dataset and apply the necessary transformations for use in training and testing machine learning models.