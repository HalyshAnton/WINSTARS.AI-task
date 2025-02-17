import json

import numpy as np
import torch
from torch import nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn.functional as F
import PIL


class TransformDataset(Dataset):
    """
    A custom dataset wrapper that applies
    transformation to images

    Args:
        dataset (torch.utils.data.Dataset):
            original dataset

        transform (callable):
            function/transform to apply to the images
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]

        X_prep = self.transform(X)

        return X_prep, y


class AnimalClassifier:
    def __init__(self, **kwargs):
        """
        Initializes the AnimalClassifier model

        Args:
            epochs (int, optional): number of training epochs (default is 10)
            batch (int, optional): batch size for training (default is 32)
            lr (float, optional): learning rate for the optimizer (default is 1e-4)
        """
        super().__init__()

        self.model = self.__get_model()
        self.dropout = nn.Dropout(0.2)
        self.features_dim = self.model.classifier[0].in_features

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch', 32)
        self.lr = kwargs.get('lr', 1e-3)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3517, 0.3557, 0.3570],
                                 std=[0.2325, 0.2347, 0.2353])
        ])

        self.labels = None

    @ staticmethod
    def __get_model():
        """
        Loads a pre-trained VGG19 model with batch normalization.

        Returns:
            torch.nn.Module:
                pre-trained VGG19 model with frozen parameters
        """
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)

        for param in vgg.parameters():
            param.requires_grad = False

        return vgg

    def train(self, dataset):
        """
        Trains the classifier using the provided dataset

        Args:
            dataset (torchvision.datasets.ImageFolder):
                dataset for training the model
        """
        self.labels = dataset.classes
        self.model.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(self.features_dim, len(self.labels))
        )

        self.model = self.model.to(self.device)

        transform_dataset = TransformDataset(dataset, self.transform)
        dataloader = DataLoader(transform_dataset,
                                batch_size=self.batch_size,
                                shuffle=True
                                )

        loss_fn = nn.CrossEntropyLoss()

        optimizer = Adam(self.model.parameters(),
                         lr=self.lr
                         )

        self.model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in dataloader:
                y_batch = y_batch.to(self.device)
                X_batch = X_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(loss.item())

    def predict_proba(self, X):
        """
        predicts animal probabilities for a given image

        Args:
            X (PIL.Image):
                input image

        Returns:
            numpy.ndarray:
                predicted class probabilities
        """
        X_prep = self.transform(X)
        X_prep = X_prep.unsqueeze(dim=0)
        X_prep = X_prep.to(self.device)

        with torch.no_grad():
            logits = self.model(X_prep)
            y_pred = F.softmax(logits, dim=-1)

        return y_pred.cpu().numpy()

    def predict(self, X):
        """
        Predicts the animal label for a given image.

        Args:
            X (PIL.Image):
                input image

        Returns:
            str:
                predicted animal label

        Raises:
            Exception: if the model is not trained.
        """
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=-1)[0]

        try:
            return self.labels[idx]
        except Exception:
            raise Exception("model isn't trained")

    def save(self, model_path, labels_path):
        """
        Saves the trained model and label mappings to files

        Args:
            model_path (str):
                file path to save the model

            labels_path (str):
                file path to save the labels
        """
        with open(labels_path, 'w') as file:
            json.dump(self.labels, file)

        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path, labels_path):
        """
        Loads a trained model and label mappings from files

        Args:
            model_path (str):
                file path of the saved model

            labels_path (str):
                file path of the saved labels
        """
        with open(labels_path) as file:
            self.labels = json.load(file)

        self.model.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(self.features_dim, len(self.labels))
        )

        weights = torch.load(model_path, weights_only=True,
                             map_location=self.device)
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)
