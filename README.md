Dobot Robotic Sorting Thesis Project

This repository contains the code and experiment analysis for an MSc thesis comparing lightweight object detection methods for robotic block sorting on a Raspberry Pi and Dobot Magician Lite.

Detection Methods

The project compares three detectors:

OpenCV HSV colour segmentation

YOLO26n

Faster R-CNN MobileNetV3-Large 320 FPN

All three detectors use the same downstream robotic sorting pipeline.

System

Raspberry Pi 4B

Dobot Magician Lite

Raspberry Pi HQ Camera

Red, yellow, blue and green blocks

Suction cup end effector

Fixed pickup height: Z = -40 mm

Blocks are sorted into two bins:

Warm bin: red and yellow

Cool bin: blue and green

Project Structure

robot_sorting/       Main robotic sorting code
dataset_tools/       Dataset annotation and visualisation scripts
Notebooks/           Dataset, training and experiment analysis notebooks
calibration/         Camera-to-robot calibration data
models/              Pretrained model files
runs/                Model outputs and physical experiment logs

Main Notebooks

dataset_split.ipynb

yolo_training.ipynb

faster_rcnn_training.ipynb

learned_model_comparison_graphs.ipynb

physical_experiment_graphs.ipynb

Experiment Pipeline

Camera image
    ↓
Object detector
    ↓
Detection centre
    ↓
Pixel-to-Dobot homography
    ↓
Target selection
    ↓
Pick and drop
    ↓
Verification and logging

Physical Experiment

Each detector was tested on the same set of physical sorting scenes across three repetitions.

The experiment logs record:

detections

selected targets

inference time

cycle time

grasp success

correct-bin result

scene completion

Results Summary

OpenCV: 100% correct-bin sorting

YOLO26n: 100% correct-bin sorting

Faster R-CNN: 96.7% correct-bin sorting

All three detectors achieved 100% physical block removal during the experiment.

Notes

The OpenCV detector was also used to generate the initial bootstrap annotations for the dataset.

The learned models were trained and evaluated using the same fixed train, validation and test split.
