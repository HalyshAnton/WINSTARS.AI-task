import argparse

import PIL
import matplotlib.pyplot as plt

from models import AnimalClassifier


def parse_args():
    """
    Parses command-line arguments for loading a trained
    model and making predictions

    arguments:
    --model-path (str): path to the trained model file. Default is 'animal_classifier.pt'.
    --labels-path (str): path to the saved animal labels. Default is 'labels.json'.
    --img (str): path to the image for prediction. Default is 'image_data/Dog/Dog-Train (3).jpeg'.
    --show (bool): whether to show the image with the prediction. Default is True.

    Returns:
        dict:
            dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Load a trained model and make predictions on an image')

    parser.add_argument("--model-path", type=str,
                        default="animal_classifier.pt",
                        help="path to trained model"
                        )

    parser.add_argument("--labels-path", type=str,
                        default="labels.json",
                        help="path to saved animal labels"
                        )

    parser.add_argument('--img', type=str, default='image_data/Dog/Dog-Train (3).jpeg',
                        help='path to image for prediction, default img.png')

    parser.add_argument('--show', type=bool, default=True,
                        help='whether to show image with prediction, default True')

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
    img = PIL.Image.open(path)

    return img

def load_model(model_path, labels_path):
    """
    Loads a trained AnimalClassifier model from the given file paths.

    Args:
        model_path (str):
            path to the trained model file

        labels_path (str):
            path to the saved label mapping file

    Returns:
        AnimalClassifier:
            loaded model ready for prediction
    """
    model = AnimalClassifier()
    model.load(model_path, labels_path)

    return model


def main():
    """
    Main function that:
    - loads a trained model
    - reads an image
    - makes a prediction
    - optionally displays the image with the predicted label
    """
    params = parse_args()

    model = load_model(params['model_path'], params['labels_path'])
    img = read_img(params['img'])

    print(type(img))

    label = model.predict(img)

    if params['show']:
        plt.title(f'Prediction: {label}')
        plt.imshow(img)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(f'ERROR: {err}')