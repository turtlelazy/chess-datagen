import torch
import os
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
cuda_available = torch.cuda.is_available()

if cuda_available:

    print("CUDA is available")

else:

    message = """
        WARNING: In order to train the model, it is advisable to use GPU.
        Change runtime type to GPU from:
        menu Runtime -> Change runtime type -> Hardware accelerator -> GPU.
        And run all the cells again.
    """
    print(message)


def serialized_model_file(
    checkpoint="best",
    use_run="train",
):
    """
        Returns the serialized file path.
    """
    return f"runs/detect/{use_run}/weights/{checkpoint}.pt"


def train(
    data,
    use_run="train",
    fallback="yolo11n.pt",
    epochs=100,
    augment=True,
    patience=50
):

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA is not available, skipping train.")
        return
    current_directory = os.getcwd()

    # Print the current working directory
    print(f"The current working directory is: {current_directory}")

    model_file = serialized_model_file("last", use_run)
    print(model_file)
    if os.path.exists(model_file):
        resume_training = True
        use_model = model_file
    else:
        resume_training = False
        use_model = fallback

    model = YOLO(
        use_model
    )

    model.train(
        data=data,
        resume=resume_training,
        epochs=epochs,
        optimizer="AdamW",
        lr0=0.0001,
        imgsz=320,
        batch=64,
        augment=augment,
        patience=patience
    )
    return model


# model = train(data="content/datasets/YOLO_07_11/data.yaml", epochs=300, use_run=None)


def test_and_confusion_matrix(model_path, data_yaml):
    # Load the trained model
    # Run prediction on the test set
    model = YOLO(model_path)

    model.val(data=data_yaml, split='test', save_json=True)


yaml_path = "content/datasets/YOLO_07_11/data.yaml"
model_path = "../models/runs/detect/train4/weights/best.pt"
# test_and_confusion_matrix(model_path, yaml_path)

# Load a random image (replace with your own image path if needed)
example_image_path = "content/datasets/YOLO_07_11/images/test/003847.png"
image = Image.open(example_image_path)

# Load the model
model = YOLO(model_path)

# Run inference
results = model(example_image_path)

# Get detections
boxes = results[0].boxes
class_names = model.names

# Create a separate image for each class detected
image_np = np.array(image)
for class_id in np.unique(boxes.cls.cpu().numpy()):
    mask = boxes.cls.cpu().numpy() == class_id
    class_boxes = boxes[mask]
    img_copy = image_np.copy()
    for box in class_boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        # Draw rectangle
        img_copy = cv2.rectangle(
            img_copy,
            (xyxy[0], xyxy[1]),
            (xyxy[2], xyxy[3]),
            (0, 255, 0),
            2
        )
        # Put class label
        cv2.putText(
            img_copy,
            class_names[int(class_id)],
            (xyxy[0], xyxy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36,255,12),
            2
        )
    output_path = f"003847_detected_class_{class_names[int(class_id)]}.png"
    Image.fromarray(img_copy).save(output_path)
    print(f"Detection image for class '{class_names[int(class_id)]}' saved to {output_path}")
