import cv2, json, os
import numpy as np


def detect_and_store(model, video_path):
    """
    Performs object detection on each frame of a video using a YOLO model and
    returns detection results in JSON format.

    Each detection includes:
        - Frame number
        - Class ID and name
        - Bounding box coordinates
        - Confidence score

    Args:
        model (YOLO): A pre-loaded YOLO model object (e.g., YOLO('yolov8s.pt')).
        video_path (str): Full path to the video file to analyze.

    Returns:
        str: A JSON-formatted string containing a list of detections across frames.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return None

    detection_data = []  # List to store all detections
    frame_number = 0  # Frame index tracker

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Finished reading video.")
            break  # Exit loop if video ends or reading fails

        # Run object detection on the current frame
        results = model(frame)

        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates and convert to int
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract confidence score and class ID
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]  # Map class ID to class name

                # Store detection information
                detection_data.append(
                    {
                        "frame": frame_number,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],  # Format: [x_min, y_min, x_max, y_max]
                    }
                )

        frame_number += 1  # Move to next frame

    cap.release()  # Release video resources
    return json.dumps(detection_data, indent=4)


def censor_objects(json_data, video_path, censored_video_path, classes_to_censor):
    """
    Applies censorship (blurring) to specific objects in a video based on detection data.

    This function:
    - Loads object detection results from a JSON string.
    - Reads the video frame-by-frame.
    - Locates the objects to be censored using bounding box data.
    - Applies Gaussian blur to those objects.
    - Writes the censored frames to a new video file.

    Args:
        json_data (str): JSON string of detection data (from detect_and_store()).
        video_path (str): Path to the input video file.
        classes_to_censor (list): List of class names to censor (e.g., ['person', 'knife']).

    Returns:
        str: Path to the saved censored video file, or None on error.
    """

    # Try to parse JSON detection data
    try:
        detections = json.loads(json_data)
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format.")
        return None

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*"mp4v")  # Video codec

    # Prepare output path
    os.makedirs(censored_video_path, exist_ok=True)
    video_name = os.path.basename(video_path)

    output_path = os.path.join(censored_video_path, video_name)

    # Set up video writer
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Finished processing video.")
            break

        # Filter detections for the current frame
        frame_detections = [d for d in detections if d["frame"] == current_frame]

        for det in frame_detections:
            if det["class_name"] in classes_to_censor:
                x1, y1, x2, y2 = det["bbox"]

                # Extract Region of Interest (ROI) and apply blur
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, (25, 25), 30)
                    frame[y1:y2, x1:x2] = blurred  # Replace original ROI

        out.write(frame)  # Write the modified frame to output
        current_frame += 1  # Advance to next frame

    # Release resources
    cap.release()
    out.release()

    print("✅ Censored video saved!")
