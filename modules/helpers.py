import os, cv2, json, shutil, random
from collections import defaultdict
from pathlib import Path


def get_video_data(video_filepath):
    """
    Loads a video file using OpenCV and prints its basic properties.

    Args:
        video_filepath (str): Relative or absolute path to the video file.

    Raises:
        IOError: If the video file cannot be opened.
    """
    # Attempt to open the video file
    cap = cv2.VideoCapture(video_filepath)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_filepath}")

    # Extract video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # Release resources
    cap.release()

    # Display video properties
    print(f"Video File: {video_filepath}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS (Frames Per Second): {fps:.2f}\n")


def extract_insights_from_detection_json_data(json_data, output_path):
    """
    Counts the occurrences of each class_name in the detection JSON data
    and saves the result in a new JSON file.

    Args:
        json_data (str): A JSON string containing a list of detections.
        output_path (str): The file path to save the resulting class count JSON.
    """
    try:
        detections = json.loads(json_data)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON data: {e}")
        return

    # Count occurrences of each class_name
    class_counts = defaultdict(int)
    for det in detections:
        class_name = det.get("class_name")
        if class_name:
            class_counts[class_name] += 1

    # Convert to regular dict and save to JSON
    with open(output_path, "w") as f:
        json.dump(dict(class_counts), f, indent=4)

    print("✅ Detection results insights saved!")


def train_model(model):
    # Get absolute path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    MODEL_NAME = "yolo_censorship_model"
    MODEL_DIR = os.path.join(script_dir, "..", "store", "custom_models", "blood")
    PATH_TO_DATASET = os.path.join(
        script_dir, "..", "store", "dataset", "combined_censored", "data.yaml"
    )

    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train the model
    model.train(
        data=PATH_TO_DATASET,
        epochs=100,  # Increase to 100 for real training
        imgsz=640,
        batch=16,
        name=MODEL_NAME,
        project=MODEL_DIR,
    )

    print(f"✅ '{MODEL_NAME}' trained successfully!")


def model_evaluation(model, file_name="model_val_metrics.json"):
    # validating model performance and getting metrics
    metrics = model.val()

    # Get the absolute path of the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path: /<script_dir>/../store/<dir_name>
    full_dir = os.path.abspath(
        os.path.join(script_dir, "..", "store", "results", file_nam)
    )
    with open(full_dir, "w") as f:
        json.dump(metrics.results_dict, f, indent=4)

    print("✅ Validation results saved to {}", file_name)


def model_prediction(model, file_name="model_predictions.json"):
    # List of image paths to predict
    image_paths = [
        "./store/dataset/blood/valid/images/0_Screenshot-2022-11-28-at-5-53-22-PM_png.rf.e1a5efecbcf8ee860fe231dbe40b8c91.jpg",
        "./store/dataset/guns/test/images/1290_jpg.rf.713bf7587f3ab0d583fab4e8543880bf.jpg",
        "./store/dataset/rifles/test/images/002e1b2ba0ec004d_jpg.rf.fbcead24b8120acb150b3f9b6ee0c356.jpg",
    ]

    # Collect results in a list
    all_predictions = []

    for path in image_paths:
        results = model.predict(source=path, save=True, conf=0.25)

        for result in results:  # Can be multiple results (e.g., video frames)
            predictions = []
            for box in result.boxes.data.tolist():  # xyxy, confidence, class
                x1, y1, x2, y2, conf, cls = box
                predictions.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": int(cls),
                        "class_name": result.names[int(cls)],
                    }
                )

            all_predictions.append({"image_path": path, "predictions": predictions})

    # Get the absolute path of the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path: /<script_dir>/../store/<dir_name>
    full_dir = os.path.abspath(
        os.path.join(script_dir, "..", "store", "results", file_name)
    )
    with open(full_dir, "w") as f:
        json.dump(all_predictions, f, indent=4)

    print("✅ All predictions saved to {}", file_name)


def combine_datasets(dataset_paths, output_path):
    """
    Combines multiple YOLO format datasets into one dataset with specified splits.

    Args:
        dataset_paths (list): List of paths to the individual datasets.
        output_path (str): Path to the output directory where the combined dataset will be saved.
    """
    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, "labels"), exist_ok=True)

    # Define class mappings: {original_dataset_index: {original_class_id: new_class_id}}
    class_mappings = {
        0: {
            0: 0,
            1: 1,
        },  # blood dataset: original class 0 → new 0 (blood), 1 → 1 (bloodstain)
        1: {0: 2, 1: 3},  # guns dataset: original class 0 → new 2 (d), 1 → 3 (gun)
        2: {0: 4},  # rifles dataset: original class 0 → new 4 (Rifle)
    }

    # Collect all image-label pairs along with their dataset origin
    image_label_pairs = []

    for dataset_idx, dataset_path in enumerate(dataset_paths):
        possible_splits = ["train", "valid", "test"]
        for split in possible_splits:
            images_dir = os.path.join(dataset_path, split, "images")
            if not os.path.exists(images_dir):
                continue

            labels_dir = os.path.join(dataset_path, split, "labels")
            if not os.path.exists(labels_dir):
                continue

            image_files = [
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            for img_file in image_files:
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_file)
                if os.path.exists(label_path):
                    image_path = os.path.join(images_dir, img_file)
                    image_label_pairs.append((image_path, label_path, dataset_idx))

    # Shuffle the list to randomize before splitting
    random.shuffle(image_label_pairs)

    # Calculate split indices
    total_images = len(image_label_pairs)
    train_end = int(0.7 * total_images)
    val_end = train_end + int(0.15 * total_images)

    # Split the list
    train_pairs = image_label_pairs[:train_end]
    val_pairs = image_label_pairs[train_end:val_end]
    test_pairs = image_label_pairs[val_end:]

    def process_pair(pair, output_split):
        """
        Processes a single image-label pair by copying the image and updating the label file.

        Args:
            pair (tuple): (image_path, label_path, dataset_idx)
            output_split (str): One of "train", "val", "test"
        """
        image_path, label_path, dataset_idx = pair
        # Copy image
        img_filename = os.path.basename(image_path)
        output_img_path = os.path.join(
            output_path, output_split, "images", img_filename
        )
        shutil.copy(image_path, output_img_path)

        # Process and copy label
        output_label_path = os.path.join(
            output_path,
            output_split,
            "labels",
            os.path.splitext(img_filename)[0] + ".txt",
        )

        # Read original label file
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Update class_ids based on dataset origin
        updated_lines = []
        mapping = class_mappings[dataset_idx]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if (
                len(parts) < 5
            ):  # Ensure it's a valid YOLO format line (class + 4 bbox coords)
                continue
            try:
                original_class_id = int(parts[0])
                if original_class_id in mapping:
                    new_class_id = mapping[original_class_id]
                    parts[0] = str(new_class_id)
                    updated_line = " ".join(parts) + "\n"
                    updated_lines.append(updated_line)
            except (ValueError, IndexError):
                continue  # Skip lines that don't conform to expected format

        # Write updated label file
        with open(output_label_path, "w") as f:
            f.writelines(updated_lines)

    # Process each pair in the splits
    for pair in train_pairs:
        process_pair(pair, "train")

    for pair in val_pairs:
        process_pair(pair, "val")

    for pair in test_pairs:
        process_pair(pair, "test")

    # Generate the new data.yaml file
    yaml_content = f"""train: ./train/images
                    val: ./val/images
                    test: ./test/images

                    nc: 5
                    names: ['blood', 'bloodstain', 'd', 'gun', 'Rifle']
                    """
    with open(os.path.join(output_path, "data.yaml"), "w") as f:
        f.write(yaml_content)


# dataset_paths = [
#     "./store/dataset/blood",
#     "./store/dataset/guns",
#     "./store/dataset/rifles",
# ]
# combine_datasets(dataset_paths, "./store/dataset/combined_censored")
