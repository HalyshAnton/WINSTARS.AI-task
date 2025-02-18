import argparse

from torchvision.datasets import ImageFolder

from models import AnimalClassifier


def parse_args():
    """
    Parses command-line arguments for training an animal classifier.

    arguments:
    --lr (float): Learning rate for training the model. Default is 1e-3.
    --epochs (int): Number of epochs for training the model. Default is 10.
    --batch (int): Batch size used for training the model. Default is 32.
    --save (bool): Whether to save the model. Default is True.
    --data-dir (str): Path to image data. Default is "image_data".
    --model-path (str): Path for saving the model. Default is "animal_classifier_custom.pt".
    --labels-path (str): Path for saving animal labels. Default is "labels_custom.json".

    Returns:
        dict:
            dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train an animal classifier.")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for training the model"
                        )

    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs for training the model"
                        )

    parser.add_argument("--batch", type=int, default=32,
                        help="batch size used for training the model"
                        )

    parser.add_argument("--save", type=bool, default=True,
                        help="whether to save the model"
                        )

    parser.add_argument("--data-dir", type=str,
                        default="image_data",
                        help="path to image data"
                        )

    parser.add_argument("--model-path", type=str,
                        default="animal_classifier_custom.pt",
                        help="path for saving the model"
                        )

    parser.add_argument("--labels-path", type=str,
                        default="labels_custom.json",
                        help="path for saving animal labels"
                        )

    args = parser.parse_args()

    return vars(args)


def load_data(root):
    """
    Loads the image dataset.

    Args:
        root (str):
            root directory of the dataset

    Returns:
        ImageFolder:
            loaded dataset
    """
    return ImageFolder(root)


def train(params):
    """
    Trains a model on the animal dataset
    using the provided parameters

    Args:
        params (dict):
            dictionary of model parameters, including
            the algorithm type, learning rate, batch size, etc

    Returns:
        AnimalClassifier
            trained model
    """
    dataset = load_data(params['data_dir'])

    model = AnimalClassifier(**params)

    model.train(dataset)

    return model


def main():
    """
    Main function that parses arguments,
    trains the model, and optionally saves it.
    """
    params = parse_args()
    model = train(params)

    if params['save']:
        model.save(params['model_path'], params['labels_path'])


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(f'ERROR: {err}')
