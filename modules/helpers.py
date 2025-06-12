import os, cv2, json
from collections import defaultdict


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


def train_single_model(model, dataset_type="blood"):
    """
    Trains a YOLO model on the specified dataset type.

    Args:
        model: The YOLO model instance (already initialized).
        dataset_type (str): One of 'blood', 'guns', or 'rifles'.

    Returns:
        None
    """

    MODEL_NAME = ""
    MODEL_DIR = ""
    PATH_TO_DATASET = ""
    
    # Get absolute path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine paths based on dataset type
    if dataset_type == "blood":
        MODEL_NAME = "blood_censor_model"
        MODEL_DIR = os.path.join(script_dir, "..", "store", "custom_models", "blood")
        PATH_TO_DATASET = os.path.join(script_dir, "..", "store", "dataset", "blood", "data.yaml")

    elif dataset_type == "guns":
        MODEL_NAME = "guns_censor_model"
        MODEL_DIR = os.path.join(script_dir, "..", "store", "custom_models", "guns")
        PATH_TO_DATASET = os.path.join(script_dir, "..", "store", "dataset", "guns", "data.yaml")

    elif dataset_type == "rifles":
        MODEL_NAME = "rifles_censor_model"
        MODEL_DIR = os.path.join(script_dir, "..", "store", "custom_models", "rifles")
        PATH_TO_DATASET = os.path.join(script_dir, "..", "store", "dataset", "rifles", "data.yaml")

    else:
        print("❌ Please select a valid 'dataset_type': 'blood', 'guns', or 'rifles'")
        return

    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train the model
    model.train(
        data=PATH_TO_DATASET,
        epochs=1,  # Increase to 100 for real training
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
