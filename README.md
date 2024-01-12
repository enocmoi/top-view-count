# Portfolio Project: Real-time Object Detection with YOLOv7

## Overview

This project demonstrates real-time object detection using the YOLOv7 (You Only Look Once version 7) deep learning model. The implementation utilizes the OpenCV library for computer vision tasks and leverages the YOLOv7 model to detect objects within a specified region of interest. The application captures video frames from an RTSP stream, processes them using YOLOv7, and displays the detected objects in real-time.

## Features

-   **Object Detection**: Utilizes the YOLOv7 model to detect objects in each video frame.
-   **Polygonal Region of Interest (ROI)**: Defines a polygonal region on the video frames where object detection is performed.
-   **Webhook Integration**: Sends the count of detected objects to an external service via a webhook.
-   **Threading**: Uses threading for parallel execution of tasks, enhancing overall performance.

## Requirements

-   Python 3.x
-   OpenCV
-   PyTorch
-   YOLOv7 model file (`yolov7-tiny_480x640.onnx`)
-   Internet connection for webhook integration

## Usage

1.  Clone the repository and navigate to the project directory.
2.  Install the required dependencies using `pip install -r requirements.txt`.
3.  Ensure the YOLOv7 model file (`yolov7-tiny_480x640.onnx`) is present in the specified path.
4.  Run the project using `python webcam_object_detection.py`.

## Configuration

-   **Webhook Endpoint**: Modify the `send_bin` function to include the correct webhook endpoint URL.
-   **Video Stream**: Adjust the RTSP stream URL in the `cv2.VideoCapture` initialization to your specific camera configuration.
-   **Region of Interest (ROI)**: Modify the `pts` array to define the polygonal ROI coordinates.

## Project Structure

-   **`webcam_object_detection.py`**: The main script containing the implementation of real-time object detection.
-   **`models/`**: Directory containing the YOLOv7 model file.
-   **`results/`**: Directory for storing the JSON results file.

## Acknowledgments

-   YOLOv7: https://github.com/WongKinYiu/yolov7
-   OpenCV: [https://opencv.org/](https://opencv.org/)
-   PyTorch: [https://pytorch.org/](https://pytorch.org/)
