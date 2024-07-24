from ultralytics import YOLO
import os
import onnx
from onnx_tf.backend import prepare


def export_model(pt_path):

    # Load a model
    model = YOLO(pt_path)  # load a pretrained model (recommended for training)

    print('\nExport to ONNX')
    model.export(format='onnx')  # export the model to ONNX format




    model_path = pt_path[:-3] + '.onnx'
    onnx_model = onnx.load(model_path)


    tf_rep = prepare(onnx_model)

    tf_rep.export_graph("./best_saved_model")


    print('\nExport to TF')
    #model.export(format='saved_model', imgsz=320, batch=1,nms=False)  # export the model to ONNX format

    #print('\nExport to tflite')
    #model.export(format='tflite', imgsz=320, batch=1, nms=False)  # export the model to ONNX format


if __name__ == '__main__':

    files_to_export = [
        '../../../computer-vision-models/object_detection/ultralytics_v8/person_detection_coco/yolov8_train38/weights/best.pt'
    ]

    for fn in files_to_export:
        assert os.path.exists(fn)
        export_model(fn)

