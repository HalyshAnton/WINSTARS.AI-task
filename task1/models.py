from abc import ABC, abstractmethod


from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class MnistRandomForest(MnistClassifierInterface):
    """
    A Random Forest classifier for the MNIST dataset

    This class implements a classifier using the RandomForestClassifier
    from scikit-learn. It preprocesses the MNIST images by flattening them
    before training and prediction
    """

    def __init__(self, **kwargs):
        """
        Initializes the MnistRandomForest classifier

        Args:
            estimators (int, optional): number of trees in the forest (default: 100)
            max_depth (int, optional): maximum depth of the trees (default: None)
        """
        self.model = RandomForestClassifier(
            n_estimators=kwargs.get('estimators', 100),
            max_depth=kwargs.get('max_depth', None)
        )

    def train(self, X, y):
        """
        Trains the Random Forest classifier

        Args:
            X (numpy.ndarray): training data of shape (N, 1, 28, 28)
            y (numpy.ndarray): training labels of shape (N)

        Returns:
            None
        """
        X_prep = self.__preprocess(X)
        self.model.fit(X_prep, y)

    def predict(self, X):
        """
        Predicts digit probabilities for the given input data

        Args:
            X (numpy.ndarray): input data of shape (N, 1, 28, 28)

        Returns:
            numpy.ndarray: digit probabilities of shape (N, 10)
        """
        X_prep = self.__preprocess(X)
        return self.model.predict_proba(X_prep)

    @staticmethod
    def __preprocess(X):
        """
        Preprocesses the input images by flattening them

        Args:
            X (numpy.ndarray): input data of shape (N, 1, 28, 28)

        Returns:
            numpy.ndarray: flattened input data of shape (N, 784)
        """
        return X.reshape(-1, 28 * 28)


class MnistFeedForward(MnistClassifierInterface):
    """
    A feedforward neural network for MNIST classification

    This model consists of a simple fully connected architecture with
    batch normalization and ReLU activations
    """

    def __init__(self, **kwargs):
        """
        Initializes the MnistFeedForward model

        The model consists of three fully connected layers with ReLU activations
        and batch normalization

        Args:
            - epochs (int, optional): number of training epochs (default: 10)
            - batch (int, optional): batch size for training (default: 32)
            - lr (float, optional): learning rate for the optimizer (default: 1e-2)
        """
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch', 32)

        self.optimizer = Adam(self.model.parameters(),
                              lr=kwargs.get('lr', 1e-2)
                              )

    def train(self, X, y):
        """
        Trains the model on the given dataset.

        Args:
            X (numpy.ndarray): training data of shape (N, 1, 28, 28)
            y (numpy.ndarray): training labels of shape (N)
        """
        X_prep = self.__preprocess(X)

        dataset = list(zip(X_prep, y))
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True
                                )

        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in dataloader:
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def predict(self, X):
        """
        Predicts digit probabilities for the given input.

        Args:
            X (numpy.ndarray): input data of shape (N, 1, 28, 28)

        Returns:
            numpy.ndarray: digit probabilities of shape (N, 10)
        """
        X_prep = self.__preprocess(X)
        X_prep = X_prep.to(self.device)

        self.model.eval()
        with torch.no_grad():
            return self.model(X_prep).cpu().numpy()

    def __preprocess(self, X):
        """
        Preprocesses input data by reshaping and converting it to a tensor

        Args:
            X (numpy.ndarray): input data of shape (N, 1, 28, 28)

        Returns:
            torch.Tensor: preprocessed input of shape (N, 784)
        """
        return torch.tensor(X.reshape(-1, 28*28),
                            dtype=torch.float32,
                            device=self.device
                            )


class MnistCNN(MnistClassifierInterface):
    """
    Convolutional Neural Network model for classifying MNIST images

    The model consists of:
        - Two convolutional layers with ReLU activation, batch normalization, and max pooling
        - A fully connected layer with dropout
        - A final softmax output layer for classification
    """

    def __init__(self, epochs=10, batch=32, lr=1e-4):
        """
        Initializes the MnistCNN model

        Args:
            epochs (int, optional): number of training epochs (default is 10)
            batch (int, optional): batch size for training (default is 32)
            lr (float, optional): learning rate for the optimizer (default is 1e-4)
        """
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        self.epochs = epochs
        self.batch_size = batch

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr
                                          )

    def train(self, X, y):
        """
        Trains the model on the provided training data

        Args:
            X (numpy.ndarray): input training data of shape (N, 1, 28, 28)
            y (numpy.ndarray): labels for the training data of shape (N,)
        """
        X_prep = self.__preprocess(X)

        dataset = list(zip(X_prep, y))
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True
                                                 )

        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in dataloader:
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def predict(self, X):
        """
        Makes predictions on the provided input data

        Args:
            X (numpy.ndarray): input data of shape (N, 1, 28, 28)

        Returns:
            numpy.ndarray: digit probabilities of shape (N, 10)
        """
        X_prep = self.__preprocess(X)
        X_prep = X_prep.to(self.device)

        self.model.eval()
        with torch.no_grad():
            return self.model(X_prep).cpu().numpy()

    def __preprocess(self, X):
        """
        Convert data to tensor

        Args:
            X (numpy.ndarray): input data of shape (N, 1, 28, 28)

        Returns:
            torch.Tensor: processed data
        """
        return torch.tensor(X,
                            dtype=torch.float32,
                            device=self.device
                            )
