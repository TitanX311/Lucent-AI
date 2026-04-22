# YOLO Object Detection Project

A custom YOLOv8 object detection model with a Streamlit inference interface.

## Features

- **Object Detection**: Detects 3 custom object classes
- **Model Training**: Train YOLOv8 models with custom data
- **Web Interface**: Streamlit-based UI for image inference
- **Bounding Box Visualization**: Annotated detection results

## Requirements

- Python 3.8+
- ultralytics (YOLOv8)
- torch/torchvision
- streamlit
- opencv-python (cv2)
- Pillow
- numpy

## Installation

```bash
pip install ultralytics torch torchvision streamlit opencv-python pillow numpy
```

## Usage

### Training

Run the training script:

```bash
python train.py
```

This will:
- Train a YOLOv8m model on your custom dataset
- Use CUDA if available, otherwise fall back to CPU
- Save results to `runs/detect/yolov8m_custom`
- Training configuration:
  - Epochs: 100
  - Batch size: 16
  - Image size: 640x640

### Inference Interface

Launch the Streamlit web interface:

```bash
streamlit run interface.py
```

Then:
1. Open your browser to the displayed URL (usually `http://localhost:8501`)
2. Upload an image
3. View detections with bounding boxes and confidence scores

## Dataset

The project expects:
- **Training images**: `data/images/train/`
- **Validation images**: `data/images/val/`
- **Labels**: Format specified in `data.yaml`
- **Classes**: 3 classes (0, 1, 2)

Edit `data.yaml` to update class names and paths.

## Model Checkpoints

- `yolov8m.pt` - Medium model weights (used for training)
- `yolov8n.pt` - Nano model weights (available)
- `runs/detect/yolov8m_custom-2/weights/best.pt` - Best trained model

## Utilities

- **split.py**: Splits dataset into train/validation sets
- **convert.py**: Converts annotation formats

## GPU Acceleration

The project automatically detects and uses CUDA if available. To force CPU usage, modify `train.py`:

```python
device = "cpu"
```

## Notes

- Confidence threshold for inference: 0.5
- Input image formats: JPG, JPEG, PNG
- Model output includes: bounding boxes, class labels, confidence scores
