import json

import torch
from torchcrf import CRF
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


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
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

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
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.labels))
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
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.labels))
        )

        weights = torch.load(model_path, weights_only=True,
                             map_location=self.device)
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)


class NERDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_len=40):
        """
        PyTorch Dataset for Named Entity Recognition (NER)

        Args:
            texts (sequence[str]):
                list of sentences

            tags (sequence[sequence[str]]):
                corresponding sequence of tag sequences
                for each sentence

            tokenizer (Tokenizer):
                tokenizer for text preprocessing

            max_len (int, optional):
                maximum length of tokenized sequences, default 40
        """
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

        unique_tags = set()
        for tags in tags:
            unique_tags |= set(tags)

        self.unique_tags = None
        self.tags_to_ids = None
        self.set_tags(list(unique_tags))

    def get_tag(self, idx):
        """Returns the tag corresponding to a given index."""
        return self.unique_tags[idx]

    def set_tags(self, tags):
        """Sets the tag-to-index mapping, ensuring 'O' is the first tag."""
        self.unique_tags = tags

        self.unique_tags.remove('O')
        self.unique_tags.insert(0, 'O')

        self.tags_to_ids = {tag: i for i, tag in enumerate(self.unique_tags)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]

        return self.transform(text, tags)

    def transform(self, text, tags=None):
        """
        Tokenizes the input text and aligns the corresponding tags
        if provided

        Args:
            text (str):
                input sentence
            tags (sequence[str]):
                corresponding tags for each word in the sentence

        Returns:
            input_ids (torch.Tensor):
                tensor of shape (1, max_len) with tokens ids
            attention_mask (torch.Tensor):
                tensor of shape (1, max_len) with 0 and 1
            aligned_tags (torch.Tensor):
                tensor of shape (1, max_len) with tags ids
        """
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        aligned_tags = self.transform_tags(encoding, tags)

        return input_ids, attention_mask, aligned_tags

    def transform_tags(self, encoding, tags):
        """
        Aligns token-level tags with subword tokenization

        Args:
            encoding:
                tokenizer output
            tags (sequence[str]):
                original word-level tags

        Returns:
            torch.Tensor:
                aligned tag sequence for subwords
        """
        if tags is None:
            return None

        word_ids = encoding.word_ids()
        aligned_tags = []
        tag_num = -1
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_tags.append(0)
            elif word_idx != prev_word_idx:
                tag_num += 1
                tag = tags[tag_num]
                tag_id = self.tags_to_ids.get(tag, 0)
                aligned_tags.append(tag_id)
            else:
                tag = tags[tag_num]
                tag_id = self.tags_to_ids.get(tag, 0)
                aligned_tags.append(tag_id)

            prev_word_idx = word_idx

        return torch.tensor(aligned_tags, dtype=torch.long)


class NERBert(nn.Module):
    def __init__(self, num_tags):
        """
        Named Entity Recognition (NER) model
        using BERT with a CRF layer

        Args:
            num_tags (int):
                number of unique entity tags
        """
        super().__init__()
        self.num_tags = num_tags

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, mask, tags=None):
        """
        Forward pass of the model

        Args:
            input_ids (torch.Tensor):
                tensor of shape (batch, max_len) with tokens ids
            mask (torch.Tensor):
                tensor of shape (batch, max_len) with 0 and 1
            tags (torch.Tensor):
                tensor of shape (batch, max_len) with tags ids

        Returns:
            If tags provided
            torch.Tensor: negative loglikelihood from crf

            if tags not provided
            list: predicted sequences of tags ids
        """
        outputs = self.bert(input_ids, attention_mask=mask)
        emissions = self.classifier(outputs.last_hidden_state)

        if tags is not None:
            return -self.crf(emissions, tags,
                             mask=mask.bool(),
                             reduction="mean")
        else:
            return self.crf.decode(emissions, mask=mask.bool())


class NER:
    def __init__(self, num_tags=22, **kwargs):
        """
        Wrapper class for training and inference of the NERBert model

        Args:
            num_tags (int):
                number of entity labels
            epochs (int, optional):
                number of training epochs (default is 10)
            batch (int, optional):
                batch size for training (default is 32)
            lr (float, optional):
                learning rate for the optimizer (default is 1e-3)
        """
        self.num_tags = num_tags
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = NERBert(num_tags)
        self.model = self.model.to(self.device)

        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch', 32)
        self.lr = kwargs.get('lr', 1e-3)

        self.dataset = None

    def train(self, dataset):
        """
        Trains the NER model

        Args:
            dataset (NERDataset): training dataset
        """
        self.dataset = dataset

        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True
                                )

        optimizer = Adam(self.model.parameters(),
                         lr=self.lr
                         )

        self.model.train()

        for _ in range(self.epochs):
            for input_ids, mask, tags in dataloader:
                input_ids = input_ids.to(self.device)
                mask = mask.to(self.device)
                tags = tags.to(self.device)

                loss = self.model(input_ids, mask, tags)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, text):
        """
        Predicts named entity tags for a given text

        Args:
            text (str): input sentence

        Returns:
            list of str: predicted tags
        """
        if self.dataset is None:
            raise Exception("model isn't trained")

        input_ids, mask, _ = self.dataset.transform(text)

        input_ids = input_ids.unsqueeze(0)
        mask = mask.unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            mask = mask.to(self.device)

            preds = self.model(input_ids, mask)
            preds = preds[0]

            tags = [self.dataset.get_tag(idx) for idx in preds]

            return tags

    def save(self, model_path, tags_path):
        """
        Saves the trained model and tags mappings to files

        Args:
            model_path (str):
                file path to save the model
            tags_path (str):
                file path to save the tags
        """
        torch.save(self.model.state_dict(), model_path)

        with open(tags_path, 'w') as file:
            json.dump(self.dataset.unique_tags, file)

    def load(self, model_path, tags_path):
        """
        Loads a trained model and tags mappings from files

        Args:
            model_path (str):
                file path of the saved model
            tags_path (str):
                file path to the saved tags
        """
        weights = torch.load(model_path, weights_only=True,
                             map_location=self.device)
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)

        with open(tags_path) as file:
            tags = json.load(file)

        dataset = NERDataset(['it'], ['O'], tokenizer)
        dataset.set_tags(tags)

        self.dataset = dataset


class ImageTextModel:
    """
    A model that integrates an image classifier and a named entity recognition (NER) model
    to compare image labels with extracted text labels.
    """
    def __init__(self, image_classifier_path,
                 image_labels_path,
                 ner_path,
                 tags_path
                 ):
        """
       Initializes the ImageTextModel with a
       pre-trained image classifier and NER model

       Args:
           image_classifier_path (str):
                path to the image classifier model file
           image_labels_path (str):
                path to the label mapping for the image classifier
           ner_path (str):
                path to the NER model file
           tags_path (str):
                path to the label mapping for the NER model
       """
        self.image_classifier = AnimalClassifier()
        self.image_classifier.load(image_classifier_path,
                                   image_labels_path)

        self.ner_model = NER()
        self.ner_model.load(ner_path, tags_path)

    def compare(self, img, text):
        """
        Compares the predicted label from image classifier
        with the extracted entity from the NER model

        Args:
            img (PIL.Image):
                input to be classified
            text (str):
                text input to be analyzed for named entities

        Returns:
            bool: True if the image label matches
            the extracted text entity, False otherwise
        """
        label = self.image_classifier.predict(img)
        tags = self.ner_model.predict(text)

        label = label.lower()

        tags = [tag[2:] for tag in tags if tag not in ('O', 'NOT_ANIMAL')]

        if tags:
            tag = tags[0].lower()
        else:
            tag = 'NOT_ANIMAL'

        return label == tag
