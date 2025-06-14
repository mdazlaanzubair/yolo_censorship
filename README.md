# YOLO Censorship

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This project focuses on building a robust system for **censoring explicit and harmful content** within images and videos. We achieve this by **fine-tuning a YOLO (You Only Look Once) model**, a cutting-edge object detection algorithm, to accurately identify and pinpoint such content. The model is meticulously trained on diverse datasets to recognize a wide range of problematic elements, ensuring comprehensive coverage and high detection accuracy.

Once the YOLO model identifies instances of explicit or harmful content, it provides precise **bounding box coordinates** for each detected item. This critical information is then seamlessly passed to a straightforward **Python script**. This script leverages the provided coordinates to automatically apply **blurred boxes** over the identified regions, effectively obscuring the problematic content while preserving the surrounding visual information. This automated approach offers a swift and efficient solution for content moderation, significantly reducing the need for manual review.

## Project Setup

To get this project up and running, follow these simple steps. Using a virtual environment is highly recommended to keep your project dependencies organized and avoid conflicts with other Python projects.

### 1. Create a Virtual Environment

Open your terminal or command prompt and run the appropriate command below to create a virtual environment named venv.

**For Windows:**

```Bash
python -m venv venv
```

**For macOS and Linux:**

```Bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

You'll need to activate the virtual environment to ensure all installed packages are isolated to this project. Choose the command that applies to your operating system:

**For Windows:**

```Bash
.\venv\Scripts\activate
```

**For macOS and Linux:**

```Bash
source venv/bin/activate
```

### 3. Install Dependencies

Once your virtual environment is active, install the necessary libraries. The easiest way is to use the requirements.txt file (if provided). Use the command appropriate for your system:

**For Windows:**

```Bash
pip install -r requirements.txt
```

**For macOS and Linux:**

```Bash
pip3 install -r requirements.txt
```

If you don't have a requirements.txt file or prefer to install them individually, you can use the following commands instead:

**For Windows:**

```Bash
pip install opencv-python-headless pandas ultralytics numpy
```

**For macOS and Linux:**

```Bash
pip3 install opencv-python-headless pandas ultralytics numpy
```

## Usage

### Step 1 - Store Creation

Create a `store` directory in the root folder cause it will contain following necessary folders:

- **`custom_models`**: It contains newly trained YOLO models after training.
- **`dataset`**: It contains all the datasets that are used in training.
- **`results`**: It contains following files:
  - `detection_results_insights.json`: It contains the insights of `detection_results.json`, like which object class appears how many times.
  - `detection_results.json`: It contains the information and annotation of detected objects in a video frame-by-frame.
  - `predictions.json`: File containing model prediction results.
- **videos**: This folder contains the videos to be user in object detection, and a folder containing censored videos.
  - `censored`: It contains only videos that are being censored based on the information given by model in `detection_results.json`.

### Step 2 - Download and Unzip Dataset

1. Download After setting up virtual environment and installing necessary dependencies, download the dataset from [here](https://doi.org/10.6084/m9.figshare.29320718.v1).
2. After downloading `unzip` the dataset in the `dataset` folder.

### Step 3 - Run the `main.py` Script

1. Make sure you have the `videos` and `dataset` setup in the right folders as documented above.

2. Check all the paths and variables in the `main.py` script are as per your folder structure.

3. Proceed with running the script

**For Windows:**

```Bash
python main.py
```

**For macOS and Linux:**

```Bash
python3 main.py
```

## Citation

#### If you utilize this repository or its contents in your research, please cite our work as follows:

```
[Citation details will be added after publication of the paper]
```

## License

This project is licensed under the **MIT License**. Please refer to the [**LICENSE**](https://opensource.org/license/MIT) for details.

## Contact
For any questions, suggestions, or issues, please open an issue in this repository or contact mdazlaan1996@gmail.com.
