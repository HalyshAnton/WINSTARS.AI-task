from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class MnistRandomForest(MnistClassifierInterface):
    """
    A Random Forest classifier for the MNIST dataset.

    This class implements a classifier using the RandomForestClassifier from scikit-learn.
    It preprocesses the MNIST images by flattening them before training and prediction.
    """

    def __init__(self, **kwargs):
        """
        Initializes the MnistRandomForest classifier.

        Args:
            estimators (int, optional): Number of trees in the forest (default: 100).
            max_depth (int, optional): Maximum depth of the trees (default: None).
        """
        self.model = RandomForestClassifier(
            n_estimators=kwargs.get('estimators', 100),
            max_depth=kwargs.get('max_depth', None)
        )

    def train(self, X, y):
        """
        Trains the Random Forest classifier.

        Args:
            X_train (numpy.ndarray): Training data of shape (N, 1, 28, 28).
            y_train (numpy.ndarray): Labels corresponding to the training data.

        Returns:
            None
        """
        X_prep = self.__preprocess(X)
        self.model.fit(X_prep, y)

    def predict(self, X):
        """
        Predicts the class probabilities for the given input data.

        Args:
            X (numpy.ndarray): Input data of shape (N, 1, 28, 28).

        Returns:
            numpy.ndarray: Predicted class probabilities of shape (N, num_classes).
        """
        X_prep = self.__preprocess(X)
        return self.model.predict_proba(X_prep)

    @staticmethod
    def __preprocess(X):
        """
        Preprocesses the input images by flattening them.

        Args:
            X (numpy.ndarray): Input data of shape (N, 1, 28, 28).

        Returns:
            numpy.ndarray: Flattened input data of shape (N, 784).
        """
        return X.reshape(-1, 28 * 28)
