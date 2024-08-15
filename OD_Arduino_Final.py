import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
from picamera2 import Picamera2, MappedArray, Preview
import serial

'''
The majority of the object detection code below was provided by the tutorial made by Edje Electronics https://www.youtube.com/@EdjeElectronics
His videos were extremely helpful, as was his GitHub page: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
The changes that I have made are with respect to functions required to send information via serial connection with an arduino that controls the servos as
they track the object. 
'''

'''
The videostream class sets up the Picamera2 video stream that the tensorflow lite model will run on.
It has parameters for setting up the resolution and max frame rate, though unless you are running an
extremely good setup the object detection model will struggle to get anywhere near 30 FPS. This is 
especially so for a Raspberry Pi without a TPU. This portion of the code was helped along by this
source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
'''

class VideoStream:
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

# Importing libraries for tflite implementation
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter 

# Initiallizing the interpreter
def initialize_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Getting object detection labels
def get_labels(path_to_labels):
    with open(path_to_labels, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])
    return labels

# Calculating the framerate, important when changing the code to isolate what is speeding/slowing the program
def calculate_frame_rate(t1, freq):
    t2 = cv2.getTickCount()
    time_elapsed = (t2 - t1) / freq
    return 1 / time_elapsed if time_elapsed > 0 else 0

# Taking the coordinates of the box around the object detected and using serial communication to send the data to the arduino
def send_coordinates(ser, x, y, center_margin_x, center_margin_y, center_frame_x, center_frame_y):
    dist_x = abs(x - center_frame_x)
    dist_y = abs(y - center_frame_y)
    if dist_x > center_margin_x or dist_y > center_margin_y:
        ser.write(f"{x},{y}\n".encode())

def main():
    # Initiallize serial communication with the arduino
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1.0)
    time.sleep(3)  

    # Commands from the command line being parsed to get the different information stored locally
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

    # Path setup and interpreter initiallization, floating or quantized determination, index-box-class score output
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
    labels = get_labels(PATH_TO_LABELS)
    interpreter = initialize_interpreter(PATH_TO_CKPT)

    # Retrieving the input output details from the model
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

    # Initialize video stream and determine the frame margins for centering data
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)

    center_margin_x = imW // 10
    center_margin_y = imH // 10
    center_frame_x = imW // 2
    center_frame_y = imH // 2

    # Capturing and processing frame loop, normalizing the input data if it's a floating model, running inference on the model 
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

        # Getting results, drawing boxes, sending coordinates loop
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

        # Frame rate display and calculation, showing the frame, exit condition and cleanup
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        frame_rate_calc = calculate_frame_rate(t1, freq)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()
    ser.close()

# Exectuting main
if __name__ == '__main__':
    main()
