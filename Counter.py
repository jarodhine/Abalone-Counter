import sys
import warnings
import numpy as np
import cv2
import torch
import torchvision  # Used by loaded model
import matplotlib.pyplot as plt

from typing import Tuple, Dict
from PIL import Image

plt.rcParams['interactive'] == True  # Use to display images

detection_threshold = 0.905
input_image_path = ""
result_boxes = []


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


def get_boxes(model_output):
    model_output = model_output[0]
    boxes = []

    pred_boxes = model_output['pred_boxes']
    scores = model_output['scores']

    for x in range(0, len(pred_boxes)):
        if scores[x] >= detection_threshold:
            boxes.append(pred_boxes[x])

    return boxes


def display_boxes_on_image(image_path, bounding_boxes):
    # Load
    image = Image.open(image_path)

    # Transforms
    image_resized = image.resize((1333, 1333))
    im_array = np.array(image_resized)
    image_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

    for x in range(0, len(bounding_boxes)):
        start_point = (int(bounding_boxes[x][0]), int(bounding_boxes[x][1]))
        end_point = (int(bounding_boxes[x][2]), int(bounding_boxes[x][3]))
        cv2.rectangle(image_bgr, start_point, end_point, (0, 0, 0), 2)

    cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection Results', 600, 600)
    cv2.imshow('Detection Results', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def inference():
    # Load Model
    loaded_model = torch.jit.load('model.ts')

    # Run inference on image
    output = loaded_model(get_input_image(input_image_path))

    # Parse results
    bounding_boxes = get_boxes(output)

    # Draw predictions on image
    display_boxes_on_image(image_path=input_image_path, bounding_boxes=bounding_boxes)

    # Display count
    print("Predicted Count: " + str(get_count(output)))


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)  # Disable Deprecation Warnings

    try:
        input_image_path = str(sys.argv[1])
    except:
        input_image_path = "TestImages/5.JPG"

    inference()
