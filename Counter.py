import warnings
import numpy as np
import cv2
import torch
import torchvision  # Used by loaded model
import tkinter
import tkinter.filedialog

from typing import Tuple, Dict
from PIL import Image
from pathlib import Path

DETECTION_THRESHOLD = 0.90
MAX_SIZE = 800

images = []
input_tuples = []
total_count = 0
detections = []


def load_directory_images(image_directory_path):
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

    if w > h:
        longest_edge = w
    else:
        longest_edge = h

    ratio = MAX_SIZE / longest_edge

    image_resized = image.resize((int(ratio * w), int(ratio * h)))

    return image_resized


def run_inference_on_image(model, loaded_image):

    # Run inference on image
    output = model(loaded_image)

    # Parse results
    temp_boxes = get_boxes(output)

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


def load_directory():
    global images
    global input_tuples

    directory = tkinter.filedialog.askdirectory()
    images, input_tuples = load_directory_images(directory)

    label_info_status.config(text="Loaded: " + str(len(input_tuples)))


def run_inference():
    global images
    global input_tuples
    global total_count

    for i in input_tuples:
        boxes = run_inference_on_image(loaded_model, i)
        total_count += get_count(boxes)
        detections.append(boxes)

    label_info_status.config(text="Complete")

    label_info_total_display.config(text=str(total_count))
    label_info_average_display.config(text=str(total_count / len(input_tuples)))


def reset():
    global images
    global input_tuples
    global detections

    images = []
    input_tuples = []
    detections = []

    print("Reset")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)  # Disable Deprecation Warnings

    # Load Model
    loaded_model = torch.jit.load('model.ts')

    # GUI
    root = tkinter.Tk()
    root.title("Abalone Counter")
    root.geometry("800x600")

    frame_menu = tkinter.Frame(master=root, bg="red")
    frame_info = tkinter.Frame(master=root, bg="green")
    frame_image = tkinter.Frame(master=root, bg="blue")

    # Configure frame sizing
    tkinter.Grid.rowconfigure(root, index=0, weight=1)
    tkinter.Grid.columnconfigure(root, index=0, weight=1)

    tkinter.Grid.rowconfigure(root, index=1, weight=4)
    tkinter.Grid.columnconfigure(root, index=0, weight=4)

    tkinter.Grid.rowconfigure(root, index=1, weight=16)
    tkinter.Grid.columnconfigure(root, index=1, weight=16)

    frame_menu.grid(row=0, column=0, columnspan=2, sticky="nsew")
    frame_info.grid(row=1, column=0, sticky="nsew")
    frame_image.grid(row=1, column=1, sticky="nsew")

    # Menu
    button_load = tkinter.Button(master=frame_menu, text="Load Images", width=8, command=load_directory)
    button_detect = tkinter.Button(master=frame_menu, text="Detect Abalone", width=8, command=run_inference)
    button_reset = tkinter.Button(master=frame_menu, text="Reset Program", width=8)

    tkinter.Grid.rowconfigure(frame_menu, index=0, weight=1)
    tkinter.Grid.columnconfigure(frame_menu, index=0, weight=2)
    tkinter.Grid.columnconfigure(frame_menu, index=1, weight=1)
    tkinter.Grid.columnconfigure(frame_menu, index=2, weight=1)
    tkinter.Grid.columnconfigure(frame_menu, index=3, weight=1)
    tkinter.Grid.columnconfigure(frame_menu, index=4, weight=2)

    button_load.grid(row=0, column=1, sticky="nsew", padx=30, pady=20)
    button_detect.grid(row=0, column=2, sticky="nsew", padx=30, pady=20)
    button_reset.grid(row=0, column=3, sticky="nsew", padx=30, pady=20)

    # Information
    label_info_title = tkinter.Label(master=frame_info, text="Information")
    label_info_total = tkinter.Label(master=frame_info, text="Total: ")
    label_info_average = tkinter.Label(master=frame_info, text="Average: ")
    label_info_total_display = tkinter.Label(master=frame_info, text="0")
    label_info_average_display = tkinter.Label(master=frame_info, text="0")
    label_info_status = tkinter.Label(master=frame_info, text="Ready")

    tkinter.Grid.rowconfigure(frame_info, index=0, weight=0)
    tkinter.Grid.rowconfigure(frame_info, index=1, weight=0)
    tkinter.Grid.rowconfigure(frame_info, index=2, weight=0)
    tkinter.Grid.rowconfigure(frame_info, index=3, weight=100)
    tkinter.Grid.rowconfigure(frame_info, index=4, weight=0)

    tkinter.Grid.columnconfigure(frame_info, index=0, weight=1)
    tkinter.Grid.columnconfigure(frame_info, index=1, weight=1)

    label_info_title.grid(row=0, column=0, columnspan=2, sticky="new")
    label_info_total.grid(row=1, column=0, sticky="new")
    label_info_average.grid(row=2, column=0, sticky="new")
    label_info_total_display.grid(row=1, column=1, sticky="new")
    label_info_average_display.grid(row=2, column=1, sticky="new")
    label_info_status.grid(row=4, column=0, columnspan=2, sticky="sew")

    # Image
    button_previous = tkinter.Button(master=frame_image, text="Previous", width=8)
    button_next = tkinter.Button(master=frame_image, text="Next", width=8)
    label_current_image = tkinter.Label(master=frame_image, text="TODO: Test Image")

    tkinter.Grid.rowconfigure(frame_info, index=0, weight=1)
    tkinter.Grid.columnconfigure(frame_image, index=0, weight=2)
    tkinter.Grid.columnconfigure(frame_image, index=1, weight=3)
    tkinter.Grid.columnconfigure(frame_image, index=2, weight=3)
    tkinter.Grid.columnconfigure(frame_image, index=3, weight=2)

    button_previous.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
    button_next.grid(row=0, column=2, sticky="nsew", padx=20, pady=20)
    label_current_image.grid(row=1, column=1, columnspan=2, sticky="nsew", pady=30)

    root.mainloop()
