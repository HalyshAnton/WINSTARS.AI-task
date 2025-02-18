import argparse

from models import NER


def parse_args():
    """
    Parses command-line arguments for loading a trained
    model and making predictions

    arguments:
    --model-path (str): path to the trained model file. Default is 'ner_mode.pt'.
    --tags-path (str): path to the saved tags. Default is 'tags.json'.
    --text (str): sentence for prediction. Default is 'It's a cat.'

    Returns:
        dict:
            dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Load a trained model and make predictions on an image')

    parser.add_argument("--model-path", type=str,
                        default="ner_model.pt",
                        help="path to trained model"
                        )

    parser.add_argument("--tags-path", type=str,
                        default="tags.json",
                        help="path to saved tags"
                        )

    parser.add_argument('--text', type=str, default="It's a cat.",
                        help="sentence for prediction")

    args = parser.parse_args()

    return vars(args)


def load_model(model_path, tags_path):
    """
    Loads a trained NER model from the given file paths.

    Args:
        model_path (str):
            path to the trained model file

        tags_path (str):
            path to the saved tags mapping file

    Returns:
        NER:
            loaded model ready for prediction
    """
    model = NER(22)
    model.load(model_path, tags_path)

    return model


def main():
    """
    Main function that:
    - loads a trained model
    - makes a prediction
    """
    params = parse_args()

    model = load_model(params['model_path'], params['tags_path'])

    tags = model.predict(params['text'])

    print(f"Text: {params['text']}")
    print(f"Tags: {tags}")


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(f'ERROR: {err}')
