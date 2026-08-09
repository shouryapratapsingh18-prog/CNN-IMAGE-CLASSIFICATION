import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, data_dir='./data'):
    """
    Downloads the CIFAR-10 dataset and returns the Train/Test DataLoaders.
    """
    # 1. Training Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 3. Download CIFAR-10 Dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # 4. Create DataLoaders
    '''a DataLoader wraps around a Dataset object to automatically handle how data is fed into your model during training and testing.'''
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


if __name__ == '__main__':
    print("Downloading dataset, please wait a moment...")
    train_loader, test_loader, classes = get_dataloaders()
    print("\n Dataset successfully prepared!")
    print(f"Total Training Batches: {len(train_loader)}")
    print(f"Total Testing Batches: {len(test_loader)}")
    print(f"Classes: {classes}")