import argparse

import pandas as pd

from models import NERDataset, NER, tokenizer


def parse_args():
    """
    Parses command-line arguments for training an animal classifier.

    arguments:
    --lr (float): Learning rate for training the model. Default is 1e-3.
    --epochs (int): Number of epochs for training the model. Default is 10.
    --batch (int): Batch size used for training the model. Default is 32.
    --save (bool): Whether to save the model. Default is True.
    --data (str): Path to data. Default is "ner_data.csv".
    --model-path (str): Path for saving the model. Default is "ner_model_custom.pt".
    --tags-path (str): Path for saving animal labels. Default is "tags_custom.json".

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

    parser.add_argument("--data", type=str,
                        default="ner_data.csv",
                        help="path to image data"
                        )

    parser.add_argument("--model-path", type=str,
                        default="ner_model_custom.pt",
                        help="path for saving the model"
                        )

    parser.add_argument("--tags-path", type=str,
                        default="tags_custom.json",
                        help="path for saving animal labels"
                        )

    args = parser.parse_args()

    return vars(args)


def load_data(data_path):
    """
    Loads the ner dataset

    Args:
        data_path (str):
            path to the dataset file (CSV format)

    Returns:
        NERDataset:
            processed dataset ready for training
    """
    df = pd.read_csv(data_path)
    df = df.sample(10)
    df.reset_index(inplace=True)

    dataset = NERDataset(df['sentence'],
                         df['tags'].apply(eval),
                         tokenizer)

    return dataset


def train(params):
    """
    Trains the NER model using the provided parameters.

    Args:
        params (dict):
            dictionary of model parameters, see parse_args()

    Returns:
        NER:
            trained NER model
    """
    dataset = load_data(params['data'])

    model = NER(**params)

    model.train(dataset)

    return model


def main():
    """
    Main function that parses arguments,
    trains the model, and optionally saves it
    """
    params = parse_args()
    model = train(params)

    print(model.predict('hello, its pretty cat'))

    if params['save']:
        model.save(params['model_path'], params['tags_path'])


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(f'ERROR: {err}')
