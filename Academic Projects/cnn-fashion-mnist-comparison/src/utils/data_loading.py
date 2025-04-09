def load_fashion_mnist_data(root="data"):
    """
    Load the FashionMNIST dataset and split it into training, validation, and test sets.

    Parameters:
    root (str): The directory where the dataset will be stored.

    Returns:
    tuple: A tuple containing the training dataset, validation dataset, and test dataset.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import random_split

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_data = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    train_size = int(0.8 * len(full_train_data))
    val_size = len(full_train_data) - train_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, batch_size=32):
    """
    Create data loaders for the training, validation, and test datasets.

    Parameters:
    train_data (Dataset): The training dataset.
    val_data (Dataset): The validation dataset.
    test_data (Dataset): The test dataset.
    batch_size (int): The number of samples per batch.

    Returns:
    tuple: A tuple containing the training, validation, and test data loaders.
    """
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader