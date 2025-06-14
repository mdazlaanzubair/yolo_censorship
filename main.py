import json
from collections import defaultdict
from ultralytics import YOLO
from modules.helpers import *
from modules.video_processing import *

# YOLO v2 variants - YOLO12n, YOLO12s, YOLO12m,	YOLO12l, YOLO12x
# STEP 1 - INITIALIZING YOLO MODEL
model = YOLO("yolo12x.pt")
print("✅ Model successfully loaded!")


# STEP 2 - TRAINING MODEL ON CUSTOM DATABASE
train_model(model)

# STEP 3 - VALIDATING MODEL
model_evaluation(model, "validation_metrics.json")


# STEP 4 - PREDICTING IMAGES
model_prediction(model, "predictions.json")


# STEP 5 - PROCESSING VIDEO USING THE MODEL, EXTRACTING AND SAVING DETECTED OBJECT INFORMATION
# E.g. class identified, bounding box, confidence, frame number
video_path = "./store/videos/mma_fight.mp4"

# INFORMATION SCRIPT
get_video_data(video_path)

detection_json_output = detect_and_store(model, video_path)
if detection_json_output:
# You can save the JSON output to a file if needed
with open('./store/results/detection_results.json', 'w') as f:
    f.write(detection_json_output)
print("✅ Detection results saved!")


# STEP 6 - CENSORING OBJECTS IN THE VIDEO FRAME-BY-FRAME BASED ON CLASSES REQUIRED TO BE CENSORED
# Note: Using data collected/extracted in "STEP 5" to censor objects

# These are classes to for which this research project is being created
classes = ["blood", "bloodstain", "Rifle", "rifle", "0", "d", "gun"]

# Assuming you have run the detect_and_store function and have 'detection_results.json'
try:
    with open("./store/results/detection_results.json", "r") as f:
        json_data = f.read()
except FileNotFoundError:
    print(
        "Error: 'detection_results.json' not found. Please run the detection script first."
    )
    exit()

video_path = "./store/videos/mma_fight.mp4"
# classes_to_censor = ["blood", "bloodstain"]
classes_to_censor = ["handbag", "dog", "tie"]

# INFORMATION
output_file_path = "./store/results/detection_results_insights.json"
extract_insights_from_detection_json_data(json_data, output_file_path)

censored_video_path = "./store/videos/censored"
censor_objects(json_data, video_path, censored_video_path, classes_to_censor)