from ultralytics import YOLO

model = YOLO(".../best.pt")

metrics = model.val(
    data=".../config3.yaml",
    plots=True
)

print("Confusion matrix saved in:", metrics.save_dir)
