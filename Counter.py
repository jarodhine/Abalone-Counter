import warnings
import numpy as np
import cv2
import torch
import torchvision  # Used by loaded model
import tkinter
import tkinter.filedialog
import statistics

from typing import Tuple, Dict
from PIL import Image
from PIL import ImageTk
from pathlib import Path

DETECTION_THRESHOLD = 0.90
MAX_SIZE = 800

images = []
input_tuples = []
total_count = 0
counts = []
detections = []
current_image = 0
current_image_bitmap = []
image_paths = []
image_list = []


def load_directory_images(image_directory_path):
    global image_paths
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
    loaded_images.insert(tkinter.END, str(image_path))

    # Load
    image = Image.open(image_path)

    # Transforms
    image_resized = resize_image(image)
    temp_image_array = np.array(image_resized)
    transposed_image_array = np.transpose(temp_image_array, [2, 0, 1])

    # Convert to Tuple
    dic: Dict[str, torch.Tensor] = ({"image": torch.Tensor(transposed_image_array)})
    temp_input_tensor: Tuple[Dict[str, torch.Tensor]] = (dic,)

    global image_list
    python_image = ImageTk.PhotoImage(image_resized)
    image_list.append(python_image)

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
    global counts

    for i in input_tuples:
        boxes = run_inference_on_image(loaded_model, i)
        counts.append(get_count(boxes))
        total_count += get_count(boxes)
        detections.append(boxes)

    label_info_status.config(text="Complete")

    label_info_total_display.config(text=str(total_count))

    mean = statistics.mean(counts)
    label_info_mean_display.config(text=str('%.2f' % mean))

    median = statistics.median(counts)
    label_info_median_display.config(text=str('%.2f' % median))

    mode = statistics.mode(counts)
    label_info_mode_display.config(text=str('%.2f' % mode))


def reset():
    global images
    global input_tuples
    global total_count
    global counts
    global detections
    global current_image
    global current_image_bitmap
    global image_paths
    global image_list

    images = []
    input_tuples = []
    total_count = 0
    counts = []
    detections = []
    current_image = 0
    current_image_bitmap = []
    image_paths = []
    image_list = []

    label_info_status.config(text="Reset")


def view_detection(x):
    selection = loaded_images.curselection()

    i = images[selection[0]]
    b = 0

    try:
        b = detections[selection[0]]
    except IndexError:
        print("No detections!")

    if b != 0:
        display_boxes_on_image(i, b)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)  # Disable Deprecation Warnings

    # Load Model
    loaded_model = torch.jit.load('model.ts')

    # GUI
    root = tkinter.Tk()
    root.title("Abalone Counter")
    root.geometry("800x600")

    frame_menu = tkinter.Frame(master=root)
    frame_info = tkinter.Frame(master=root, bd=2, relief=tkinter.RIDGE)
    frame_image = tkinter.Frame(master=root, bd=2, relief=tkinter.RIDGE)

    frame_menu.grid_propagate(True)
    frame_info.grid_propagate(True)
    frame_image.grid_propagate(True)

    # Configure frame sizing
    tkinter.Grid.rowconfigure(root, index=0, weight=2)
    tkinter.Grid.columnconfigure(root, index=0, weight=1)

    tkinter.Grid.rowconfigure(root, index=1, weight=4)
    tkinter.Grid.columnconfigure(root, index=0, weight=4)

    tkinter.Grid.rowconfigure(root, index=1, weight=12)
    tkinter.Grid.columnconfigure(root, index=1, weight=12)

    frame_menu.grid(row=0, column=0, columnspan=2)
    frame_info.grid(row=1, column=0, sticky="nsew")
    frame_image.grid(row=1, column=1, sticky="nsew")

    # Menu
    button_load = tkinter.Button(master=frame_menu, text="Load Images", width=16, command=load_directory)
    button_detect = tkinter.Button(master=frame_menu, text="Detect Abalone", width=16, command=run_inference)
    button_reset = tkinter.Button(master=frame_menu, text="Reset Program", width=16, command=reset)

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
    label_info_mean = tkinter.Label(master=frame_info, text="Mean: ")
    label_info_median = tkinter.Label(master=frame_info, text="Median: ")
    label_info_mode = tkinter.Label(master=frame_info, text="Mode: ")

    label_info_total_display = tkinter.Label(master=frame_info, text="0")
    label_info_mean_display = tkinter.Label(master=frame_info, text="0")
    label_info_median_display = tkinter.Label(master=frame_info, text="0")
    label_info_mode_display = tkinter.Label(master=frame_info, text="0")
    label_info_status = tkinter.Label(master=frame_info, text="Ready")

    tkinter.Grid.rowconfigure(frame_info, index=0, weight=0)
    tkinter.Grid.rowconfigure(frame_info, index=1, weight=0)
    tkinter.Grid.rowconfigure(frame_info, index=2, weight=0)
    tkinter.Grid.rowconfigure(frame_info, index=3, weight=0)
    tkinter.Grid.rowconfigure(frame_info, index=4, weight=0)
    tkinter.Grid.rowconfigure(frame_info, index=5, weight=100)
    tkinter.Grid.rowconfigure(frame_info, index=6, weight=0)

    tkinter.Grid.columnconfigure(frame_info, index=0, weight=1)
    tkinter.Grid.columnconfigure(frame_info, index=1, weight=1)

    label_info_title.grid(row=0, column=0, columnspan=2, sticky="new")

    label_info_total.grid(row=1, column=0, sticky="new")
    label_info_mean.grid(row=2, column=0, sticky="new")
    label_info_median.grid(row=3, column=0, sticky="new")
    label_info_mode.grid(row=4, column=0, sticky="new")

    label_info_total_display.grid(row=1, column=1, sticky="new")
    label_info_mean_display.grid(row=2, column=1, sticky="new")
    label_info_median_display.grid(row=3, column=1, sticky="new")
    label_info_mode_display.grid(row=4, column=1, sticky="new")

    label_info_status.grid(row=6, column=0, columnspan=2, sticky="sew")

    # Image
    scrollbar = tkinter.Scrollbar(master=frame_image, orient=tkinter.VERTICAL)
    loaded_images = tkinter.Listbox(master=frame_image, yscrollcommand=scrollbar.set, selectmode=tkinter.SINGLE)

    scrollbar.config(command=loaded_images.yview)

    tkinter.Grid.rowconfigure(frame_image, index=0, weight=1)
    tkinter.Grid.rowconfigure(frame_image, index=1, weight=10)
    tkinter.Grid.rowconfigure(frame_image, index=2, weight=1)

    tkinter.Grid.columnconfigure(frame_image, index=0, weight=1)
    tkinter.Grid.columnconfigure(frame_image, index=1, weight=10)
    tkinter.Grid.columnconfigure(frame_image, index=2, weight=1)

    scrollbar.grid(row=0, column=2, rowspan=3, sticky="ns")
    loaded_images.grid(row=1, column=1, sticky="nsew")

    loaded_images.bind('<Double-1>', view_detection)

    root.mainloop()
