import ultralytics
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = ultralytics.YOLO("yolov8m.pt")

if __name__ == "__main__":
    model.train(
        data="data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,
        name="yolov8m_custom",
    )