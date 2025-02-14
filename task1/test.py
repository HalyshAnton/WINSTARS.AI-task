import argparse
import json

import PIL
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Resize

from models import MnistClassifier


def parse_args():
    """
    Parses command-line arguments for loading a trained
    model and making predictions

    arguments:
    --algorithm (str): the model type, one of ('rf', 'nn', 'cnn'). default is 'rf'
    --model-path (str): path to the trained model file. default is 'mnist_rf.pkl'
    --img (str): path to the image for prediction. default is 'img.png'
    --save-path (str): path to save the prediction results as a JSON file. default is 'result.json'
    --show (bool): whether to show the image with the prediction. default is True

    Returns:
        dict
            dictionary containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Load a trained model and make predictions on an image')

    parser.add_argument('--algorithm', type=str, choices=['rf', 'nn', 'cnn'],
                        default='rf', help='model type, one of (rf, nn, cnn), default rf')

    parser.add_argument('--model-path', type=str, default='mnist_rf.pkl',
                        help='path to trained model file, default mnist_rf.pt')

    parser.add_argument('--img', type=str, default='img.png',
                        help='path to image for prediction, default img.png')

    parser.add_argument('--save-path', type=str, default='result.json',
                        help='path to json file with prediction, default result.json')

    parser.add_argument('--show', type=bool, default=True,
                        help='whether to show image with prediction, default True')

    args = parser.parse_args()

    return vars(args)


def read_img(path):
    """
    Reads an image from the given path and
    resizes it to (1, 1, 28, 28)

    Args:
        path (str):
            path to the image file.

    Returns:
        numpy.ndarray:
            preprocessed image of shape (1, 1, 28, 28).
    """
    transform = Resize((28, 28))

    img = PIL.Image.open(path)
    img = transform(img)
    img = np.array(img)
    img = np.expand_dims(img, axis=(0, 1))

    return img

def load_model(algorithm, path):
    """
    Loads a trained MNIST model from the given file path

    Args:
        algorithm (str):
            model type, one of ('rf', 'nn', 'cnn')
        path (str):
            path to the trained model file

    Returns:
        MnistClassifier:
            loaded model ready for prediction.
    """
    model = MnistClassifier(algorithm)
    model.load(path)

    return model


def main():
    """
    Main function that:
    - loads a trained model
    - reads an image
    - makes a prediction
    - saves the prediction to a JSON file
    - optionally displays the image with the predicted label
    """
    params = parse_args()

    model = load_model(params['algorithm'], params['model_path'])
    img = read_img(params['img'])

    pred = model.predict(img)[0]

    with open(params['save_path'], 'w') as file:
        data = {params['img']: pred.tolist()}
        json.dump(data, file)

    if params['show']:
        img = np.squeeze(img, axis=(0, 1))

        label = pred.argmax(axis=-1)

        plt.title(f'Prediction: {label}')
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(f'ERROR: {err}')