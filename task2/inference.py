import argparse

from PIL import Image
import matplotlib.pyplot as plt

from models import ImageTextModel


def parse_args():
    """
    Parses command-line arguments for loading a trained
    model and making predictions.

    Arguments:
    --text (str): Sentence for prediction. Default is "It's a cat.".
    --img (str): Path to the image for prediction. Default is 'image_data/Dog/Dog-Train (3).jpeg'.
    --classifier (str): Path to the trained image classifier model file. Default is 'animal_classifier.pt'.
    --labels-path (str): Path to the saved animal labels. Default is 'labels.json'.
    --ner (str): Path to the trained NER model file. Default is 'ner_model.pt'.
    --tags-path (str): Path to the saved tags. Default is 'tags.json'.

    Returns:
        dict: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Load a trained model and make predictions on an image and text.')

    parser.add_argument('text', type=str, default="It's a cat.",
                        help="Sentence for prediction")

    parser.add_argument('img', type=str, default='image_data/Dog/Dog-Train (3).jpeg',
                        help='Path to image for prediction')

    parser.add_argument("--classifier", type=str,
                        default="animal_classifier.pt",
                        help="Path to the trained image classifier model file")

    parser.add_argument("--labels-path", type=str,
                        default="labels.json",
                        help="Path to saved animal labels")

    parser.add_argument("--ner", type=str,
                        default="ner_model.pt",
                        help="Path to the trained NER model file")

    parser.add_argument("--tags-path", type=str,
                        default="tags.json",
                        help="Path to saved tags")

    args = parser.parse_args()

    return vars(args)


def read_img(path):
    """
    Reads an image from the given path

    Args:
        path (str):
            path to the image file

    Returns:
        PIL.Image:
            loaded image
    """
    img = Image.open(path)

    return img


def load_model(image_classifier_path, image_labels_path,
               ner_path, tags_path):
    """
    Loads a trained ImageTextModel from the given file paths

    Args:
        image_classifier_path (str):
            path to the trained image classifier model file
        image_labels_path (str):
            path to the saved label mapping file
        ner_path (str):
            path to the trained NER model file
        tags_path (str):
            path to the saved tags file

    Returns:
        ImageTextModel:
            loaded model ready for prediction.
    """
    model = ImageTextModel(image_classifier_path,
                           image_labels_path,
                           ner_path,
                           tags_path
                           )

    return model


def main():
    """
    Main function that parses arguments, loads the model,
    processes the input image and text, and show predictions.
    """
    params = parse_args()

    model = load_model(params['classifier'],
                       params['labels_path'],
                       params['ner'],
                       params['tags_path']
                       )

    img = read_img(params['img'])

    is_same = model.compare(img, params['text'])

    return is_same


if __name__ == '__main__':
    main()
