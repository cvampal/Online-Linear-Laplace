import numpy as np
import torch
from torchvision import datasets, transforms


def permutate_image_pixels(image, permutation):
    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]
        image = image.view(c, h, w)
        return image
    
    
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)
    
    
def get_permuted_MNIST(num_task=50, seed=42):
    np.random.seed(seed)
    config = {'size': 28, 'channels': 1, 'classes': 10} 
    MNIST_trainset = datasets.MNIST(root='data/', train=True, download=True,
                                    transform=transforms.ToTensor())
    MNIST_testset = datasets.MNIST(root='data/', train=False, download=True,
                                transform=transforms.ToTensor())
    permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(num_task-1)]
    train_datasets = []
    test_datasets = []
    for perm in permutations:
        train_datasets.append(TransformedDataset(
            MNIST_trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
        ))
        test_datasets.append(TransformedDataset(
            MNIST_testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
        ))
    return train_datasets, test_datasets

# define a function to get fashing mnist dataset
def get_fashionMNIST(seed=42):
    np.random.seed(seed)
    config = {'size': 28, 'channels': 1, 'classes': 10}
    fashionMNIST_trainset = datasets.FashionMNIST(root='data/', train=True, download=True,
                                    transform=transforms.ToTensor())
    fashionMNIST_testset = datasets.FashionMNIST(root='data/', train=False, download=True,
                                transform=transforms.ToTensor())
    # add padding to make the size 32x32
    padding = transforms.Pad(2)
    train_datasets = TransformedDataset(fashionMNIST_trainset, transform=padding)
    test_datasets = TransformedDataset(fashionMNIST_testset, transform=padding)
    return train_datasets, test_datasets

def get_cifar10(seed=42):
    np.random.seed(seed)
    config = {'size': 32, 'channels': 3, 'classes': 10}
    cifar10_trainset = datasets.CIFAR10(root='data/', train=True, download=True,
                                    transform=transforms.ToTensor())
    cifar10_testset = datasets.CIFAR10(root='data/', train=False, download=True,
                                transform=transforms.ToTensor())
    return cifar10_trainset, cifar10_testset

def get_SVHN(seed=42):
    np.random.seed(seed)
    config = {'size': 32, 'channels': 3, 'classes': 10}
    SVHN_trainset = datasets.SVHN(root='data/', split='train', download=True,
                                    transform=transforms.ToTensor())
    SVHN_testset = datasets.SVHN(root='data/', split='test', download=True,
                                transform=transforms.ToTensor())
    return SVHN_trainset, SVHN_testset

def get_MNIST(seed=42):
    np.random.seed(seed)
    config = {'size': 28, 'channels': 1, 'classes': 10}
    MNIST_trainset = datasets.MNIST(root='data/', train=True, download=True,
                                    transform=transforms.ToTensor())
    MNIST_testset = datasets.MNIST(root='data/', train=False, download=True,
                                transform=transforms.ToTensor())
    # zero-pad the images to make the size 32x32
    padding = transforms.Pad(2)
    MNIST_trainset = TransformedDataset(MNIST_trainset, transform=padding)
    MNIST_testset = TransformedDataset(MNIST_testset, transform=padding)
    
    return MNIST_trainset, MNIST_testset

def get_combined_data():
    train_datasets = []
    test_datasets = []
    
    dataset_names = ['MNIST', 'Permuted MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    
    # get MNIST dataset
    MNIST_trainset, MNIST_testset = get_MNIST()
    train_datasets.append(MNIST_trainset)
    test_datasets.append(MNIST_testset)
    
    # get permuted MNIST dataset
    permuted_MNIST_trainset, permuted_MNIST_testset = get_permuted_MNIST(num_task=1)
    
    # zero pad the images to make the size 32x32
    padding = transforms.Pad(2)
    permuted_MNIST_trainset[0] = TransformedDataset(permuted_MNIST_trainset[0], transform=padding)
    permuted_MNIST_testset[0] = TransformedDataset(permuted_MNIST_testset[0], transform=padding)
    train_datasets.append(permuted_MNIST_trainset[0])
    test_datasets.append(permuted_MNIST_testset[0])
    
    # get FashionMNIST dataset
    fashionMNIST_trainset, fashionMNIST_testset = get_fashionMNIST()
    train_datasets.append(fashionMNIST_trainset)
    test_datasets.append(fashionMNIST_testset)
    
    # get CIFAR10 dataset
    cifar10_trainset, cifar10_testset = get_cifar10()
    train_datasets.append(cifar10_trainset)
    test_datasets.append(cifar10_testset)
    
    # get SVHN dataset
    SVHN_trainset, SVHN_testset = get_SVHN()
    train_datasets.append(SVHN_trainset)
    test_datasets.append(SVHN_testset)
    
    return train_datasets, test_datasets, dataset_names