# Dobot Robotic Sorting Thesis Project

This repository contains the code, training notebooks and experiment analysis for my MSc thesis comparing object detection methods for robotic block sorting.

## Detection Methods

- OpenCV HSV colour segmentation
- YOLO26n
- Faster R-CNN MobileNetV3-Large 320 FPN

All three detectors use the same robotic sorting pipeline.

## Project Structure

```text
robot_sorting/       Robotic sorting code
dataset_tools/       Dataset preparation scripts
Notebooks/           Training and analysis notebooks
calibration/         Camera-to-robot calibration files
models/              Model files
runs/                Training outputs and experiment logs
