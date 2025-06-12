# YOLO Censorship

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

(Instructions on how to use the project will be added here later)

## Contributing

(Information on how to contribute will be added here later)

## License

(License information will be added here later)
