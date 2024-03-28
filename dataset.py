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
        