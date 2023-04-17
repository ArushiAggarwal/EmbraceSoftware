import serial

from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2
import numpy as np
import os
import argparse

ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# Setup the PiCamera and grab a reference to the raw camera capture
camera = PiCamera() #put serial
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

# Load YOLO object detector model and labels
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
classes = []
with open("surgicaltools.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize variables for object tracking
frame_num = 0
object_trackers = {}

# Loop over the frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Grab the raw NumPy array representing the image and initialize output image
    image = frame.array
    output = image.copy()

    # Extract bounding boxes and confidence scores from YOLO model
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    boxes = []
    confidences = []
    classIDs = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5 and classID == 0:  # Only detect surgical tools
                # Scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to remove overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Loop over the remaining detection indexes after NMS and draw bounding boxes and labels
    if len(idxs) > 0:
        for i in idxs.flatten():
            if i not in object_trackers:
                # Initialize object tracker for new detection
                tracker = cv2.TrackerKCF_create()
                tracker.init(image, tuple(boxes[i]))
                object_trackers[i] = tracker
            else:
                # Update existing object tracker
                success, box = object_trackers[i].update(image)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Display the frame
    cv2.imshow("frame", frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()

