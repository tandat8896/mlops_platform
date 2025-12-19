#!/bin/bash

# This script downloads the YOLO checkpoint from Ultralytics' public repository.
# It saves the checkpoint to the specified directory.

YOLO11n="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
YOLO11s="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
YOLO11m="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"

# Create the checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download the YOLO checkpoints
wget -O checkpoints/yolo11n.pt $YOLO11n
wget -O checkpoints/yolo11s.pt $YOLO11s
wget -O checkpoints/yolo11m.pt $YOLO11m
