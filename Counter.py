import sys
import warnings
import numpy as np
import torch
import torchvision  # Used by loaded model
import matplotlib.pyplot as plt

from typing import Tuple, Dict
from PIL import Image

plt.rcParams['interactive'] == True  # Use to display images

detection_threshold = 0.905
input_image_path = ""


# Load image from file path and convert to Tuple
def get_input_image(image_path):
    # Load
    image = Image.open(image_path)

    # Transforms
    image_resized = image.resize((1333, 1333))
    im_array = np.array(image_resized)
    transposed_image_array = np.transpose(im_array, [2, 0, 1])

    # Convert to Tuple
    dic: Dict[str, torch.Tensor] = ({"image": torch.Tensor(transposed_image_array)})
    input_image: Tuple[Dict[str, torch.Tensor]] = (dic,)

    return input_image


def get_count(model_output):
    count = 0

    for x in model_output[0]['scores']:
        if x >= detection_threshold:
            count = count + 1

    return count


def inference():
    # Load Model
    loaded_model = torch.jit.load('model.ts')

    # Run inference on image
    output = loaded_model(get_input_image(input_image_path))

    # Display count
    print("Predicted Count: " + str(get_count(output)))


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)  # Disable Deprecation Warnings

    try:
        input_image_path = str(sys.argv[1])
    except:
        input_image_path = "TestImages/5.JPG"

    inference()
