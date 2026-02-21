import torch
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO("yolov8s.yaml").to(device)

results = model.train(data="config3.yaml", epochs=100, batch=4, device=device)
