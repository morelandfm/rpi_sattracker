import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
from picamera2 import Picamera2, MappedArray, Preview
import serial

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self, resolution=(640,480), framerate=30):
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = resolution
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.controls.FrameRate = framerate
        self.picam2.configure("preview")
        self.frame = None
        self.stopped = False
            
    def start(self):
        self.picam2.start()
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.frame = self.picam2.capture_array()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True  
        self.picam2.stop()

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter 

def initialize_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_labels(path_to_labels):
    with open(path_to_labels, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])
    return labels

def calculate_frame_rate(t1, freq):
    t2 = cv2.getTickCount()
    time_elapsed = (t2 - t1) / freq
    return 1 / time_elapsed if time_elapsed > 0 else 0

def send_coordinates(ser, x, y, center_margin_x, center_margin_y, center_frame_x, center_frame_y):
    dist_x = abs(x - center_frame_x)
    dist_y = abs(y - center_frame_y)
    if dist_x > center_margin_x or dist_y > center_margin_y:
        ser.write(f"{x},{y}\n".encode())

def main():
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1.0)
    time.sleep(3)  

    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', required=True, help='Folder the .tflite file is located in')
    parser.add_argument('--graph', default='detect.tflite', help='Name of the .tflite file')
    parser.add_argument('--labels', default='labelmap.txt', help='Name of the labelmap file')
    parser.add_argument('--threshold', default=0.98, help='Minimum confidence threshold for displaying detected objects')
    parser.add_argument('--resolution', default='1280x720', help='Desired webcam resolution in WxH')
    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)

    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
    labels = get_labels(PATH_TO_LABELS)
    interpreter = initialize_interpreter(PATH_TO_CKPT)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    outname = output_details[0]['name']
    if 'StatefulPartitionedCall' in outname:
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    box_width = 200
    box_height = 200

    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)

    center_margin_x = imW // 10
    center_margin_y = imH // 10
    center_frame_x = imW // 2
    center_frame_y = imH // 2

    while True:
        t1 = cv2.getTickCount()
        frame1 = videostream.read()
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                center_y = int((boxes[i][0] + boxes[i][2]) / 2 * imH)
                center_x = int((boxes[i][1] + boxes[i][3]) / 2 * imW)
                xmin = max(1, center_x - box_width // 2)
                ymin = max(1, center_y - box_height // 2)
                xmax = min(imW, center_x + box_width // 2)
                ymax = min(imH, center_y + box_height // 2)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), 
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                send_coordinates(ser, center_x, center_y, center_margin_x, center_margin_y, center_frame_x, center_frame_y)

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        frame_rate_calc = calculate_frame_rate(t1, freq)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()
    ser.close()

if __name__ == '__main__':
    main()
