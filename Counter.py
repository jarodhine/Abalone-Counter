import sys
import warnings
import numpy as np
import cv2
import torch
import torchvision  # Used by loaded model

from typing import Tuple, Dict
from PIL import Image
from pathlib import Path

DETECTION_THRESHOLD = 0.90
MAX_SIZE = 800
total_count = 0


def load_directory(image_directory_path):
    image_paths = []
    temp_images_arrays = []
    temp_input_tensors = []

    # Get file paths
    for x in Path(image_directory_path).rglob('*.jpg'):
        image_paths.append(x)
    for x in Path(image_directory_path).rglob('*.png'):
        image_paths.append(x)
    for x in Path(image_directory_path).rglob('*.bmp'):
        image_paths.append(x)

    # Load images
    for image in image_paths:
        x, y = load_image(image)
        temp_images_arrays.append(x)
        temp_input_tensors.append(y)

    return temp_images_arrays, temp_input_tensors


def load_image(image_path):
    # Load
    image = Image.open(image_path)

    # Transforms
    # image_resized = image.resize((800, 800))
    image_resized = resize_image(image)
    temp_image_array = np.array(image_resized)
    transposed_image_array = np.transpose(temp_image_array, [2, 0, 1])

    # Convert to Tuple
    dic: Dict[str, torch.Tensor] = ({"image": torch.Tensor(transposed_image_array)})
    temp_input_tensor: Tuple[Dict[str, torch.Tensor]] = (dic,)

    return temp_image_array, temp_input_tensor


def resize_image(image):
    w, h = image.size

    longest_edge = 1
    if w > h:
        longest_edge = w
    else:
        longest_edge = h

    ratio = MAX_SIZE / longest_edge

    image_resized = image.resize((int(ratio * w), int(ratio * h)))

    return image_resized


def run_inference(model, loaded_image):
    print("Processing...")

    # Run inference on image
    output = model(loaded_image)

    # Parse results
    temp_boxes = get_boxes(output)

    # Draw predictions on image
    # display_boxes_on_image(image_path=input_image_path, bounding_boxes=output_boxes)

    # Display count
    # print("Predicted Count: " + str(get_count(output_boxes)))

    return temp_boxes


def get_boxes(model_output):
    model_output = model_output[0]
    temp_boxes = []

    pred_boxes = model_output['pred_boxes']
    scores = model_output['scores']

    for x in range(0, len(pred_boxes)):
        if scores[x] >= DETECTION_THRESHOLD:
            temp_boxes.append(pred_boxes[x])

    return temp_boxes


def get_count(result_boxes):
    return len(result_boxes)


def display_boxes_on_image(image, bounding_boxes):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for x in range(0, len(bounding_boxes)):
        start_point = (int(bounding_boxes[x][0]), int(bounding_boxes[x][1]))
        end_point = (int(bounding_boxes[x][2]), int(bounding_boxes[x][3]))
        cv2.rectangle(image_bgr, start_point, end_point, (0, 0, 0), 2)

    cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection Results', 800, 600)
    cv2.imshow('Detection Results', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)  # Disable Deprecation Warnings

    try:
        input_image_path = str(sys.argv[1])
    except IndexError:
        input_image_path = "TestImages/"

    # Load images
    images, input_tuples = load_directory(input_image_path)

    # Load Model
    loaded_model = torch.jit.load('model.ts')

    detections = []

    for i in input_tuples:
        boxes = run_inference(loaded_model, i)
        total_count += get_count(boxes)
        detections.append(boxes)

    print("Predicted Total: " + str(total_count))
    print("Average per Image: " + str(total_count / len(input_tuples)))

    # Display First Detection Result
    display_boxes_on_image(images[0], detections[0])
