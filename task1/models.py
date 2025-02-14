from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass