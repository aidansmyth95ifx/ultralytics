from ultralytics import YOLO

# PRO TIP - Replace 'model=yolov5n.pt' with new 'model=yolov5nu.pt'.
model = YOLO('yolov5nu.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='ultralytics/cfg/datasets/coco_person.yaml', epochs=100, batch=32, save_period=5, imgsz=320, device=0)  # train the model
results = model.val()
results = model('https://ultralytics.com/images/zidane.jpg')  # predict on an image
results = model.export(format='saved_model')  # export the model to ONNX format
