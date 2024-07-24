
from ultralytics import YOLO
import os

model_path = '../../../computer-vision-models/object_detection/ultralytics_v8/person_detection_coco/yolov8_train38/weights/best.pt'


assert os.path.exists(model_path)
img_path = 'zidane.jpg'

model = YOLO(model_path)
print(model)
results = model(img_path)

for result in results:

    print('result: {}'.format(result))
    print(result.boxes)
    
    for idx in range(len(result.boxes)):
        print('bbox: {}'.format(result.boxes.xyxy[idx]))
        print('cls: {}'.format(result.boxes.cls[idx]))
        print('conf: {}'.format(result.boxes.conf[idx]))


