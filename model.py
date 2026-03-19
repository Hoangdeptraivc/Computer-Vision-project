from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Đường dẫn đến ảnh và model
image = r"D:\data_set\valorant_selected_frames\fn.v1i.yolov8\train\images\clip2-3-_jpg.rf.3c9172c60a9922671311ecfb635bae44.jpg"
model_path = r"D:\game\Myproject\toolvalorant\best.pt"

# Load model
model = YOLO(model_path)

# Đọc ảnh gốc
img = cv2.imread(image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Dự đoán
results = model(img)

# Vẽ kết quả lên ảnh
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            # Lấy tọa độ
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Lấy confidence và class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'

            # Vẽ bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ label
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Hiển thị ảnh
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.title('Kết quả dự đoán')
plt.show()

# In thông tin chi tiết
print("Kết quả dự đoán:")
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            print(f"{i + 1}. {class_name}: {conf:.2f}")
