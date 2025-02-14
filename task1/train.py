import argparse

import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms

from models import MnistClassifier


def parse_args():
    """
    Parses command-line arguments for training
    model on the MNIST dataset.

    arguments
    --algorithm (str): the model type, one of ('rf', 'nn', 'cnn'). default is 'rf'
    --mnist-dir (str): directory where MNIST dataset is located. if not provided, the dataset will be downloaded
    --estimators (int): number of trees for the random forest model. default is 100
    --max-depth (int): maximum depth of the trees for the random forest model, optional
    --lr (float): learning rate for training the model. default is 1e-4
    --epochs (int): number of epochs for training the model. default is 10
    --batch (int): batch size used for training the model. default is 32
    --save (bool): whether to save the model after training. default is True
    --save-path (str): path to save the trained model, optional

    Returns:
        dict
            a dictionary containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a model on MNIST dataset')

    parser.add_argument('--algorithm', type=str, choices=['rf', 'nn', 'cnn'],
                        default='rf', help='model type, one of (rf, nn, cnn)'
    )

    parser.add_argument('--mnist-dir', type=str, default=None,
                        help='dir where mnist dataset is located'
    )
    parser.add_argument('--estimators', type=int, default=100,
                        help='number of trees'
    )

    parser.add_argument('--max-depth', type=int, default=None,
                        help='max depth of trees'
    )
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate'
    )
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs'
    )
    parser.add_argument('--batch', type=int, default=32,
                        help='batch size'
    )

    parser.add_argument('--save', type=bool, default=True,
                        help='whether to save'
                        )

    parser.add_argument('--save-path', type=str, default=None,
                        help='path for model saving'
                        )

    args = parser.parse_args()

    return vars(args)


def load_data(root=None):
    """
    Loads and preprocesses the MNIST dataset.

    Args:
        root (str, optional):
            directory where the MNIST dataset is stored.
            If None, the dataset will be downloaded

    Returns:
        numpy.ndarray
            array of shape (N, 1, 28, 28) containing the MNIST images

        numpy.ndarray
            array of shape (N) containing the labels for the MNIST images
    """
    transform = transforms.ToTensor()

    if root:
        dataset = MNIST(root=root, download=False, transform=transform)
    else:
        dataset = MNIST(root='data', download=True, transform=transform)

    X = []
    y = []

    for img, label in dataset:
        img = img.numpy()

        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def train(params):
    """
    Trains a model on the MNIST dataset using the provided parameters

    Args:
        params (dict):
            dictionary of model parameters, including
            the algorithm type, learning rate, batch size, etc

    Returns:
        MnistClassifier
            trained model
    """
    X, y = load_data(params['mnist_dir'])

    model = MnistClassifier(**params)

    model.train(X, y)

    return model


def main():
    """
    Main function that parses arguments,
    trains the model, and optionally saves it.
    """
    params = parse_args()
    model = train(params)

    if params['save']:
        model.save(params['save_path'])


if __name__ == '__main__':
    main()