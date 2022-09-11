import os
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch


def prepare_data():
    # transforms for images
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # prepare transforms standard to MNIST
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_train = [mnist_train[i] for i in range(2200)]

    mnist_train, mnist_val = random_split(mnist_train, [2000, 200])

    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    mnist_test = [mnist_test[i] for i in range(3000, 4000)]

    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    # return mnist_train, mnist_val, mnist_test
    return training_data, mnist_val, mnist_test
