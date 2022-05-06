# Imports

import cv2
import numpy as np
import argparse

# This is an import for the Raspberry Pi GPIO pins

# import RPI.GPIO as GPIO


def liveDetection():

    # This here is to control the LED lights for the Raspberry Pi. So when the loop starts the red light is on in pin 4. This is because nothing has been detected yet
    #
    # GPIO.output(3, GPIO.LOW)
    # GPIO.output(4, GPIO.HIGH)

    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg",
                                default=r'C:\Users\aliya\PycharmProjects\FinalProject\darknet\cfg\yolov4-tiny-custom.cfg',
                                help="YOLO cfg file")
    parser.add_argument("--weights",
                                default=r'C:\Users\aliya\PycharmProjects\FinalProject\training\yolov4-tiny-custom_best.weights',
                                help="YOLOV4-tiny weights file")
    parser.add_argument("--labels", default='darknet/data/obj.names', help="labels file")

    args = parser.parse_args()

    # Confidence threshold means it will only show the images that YOLO is more confident than the confident threshold humber (percentage)

    confidenceThreshold = 0.5

    # NMS one is for the non-max suppression so that we don't get more than one detection for one image

    nmsThreshold = 0.5

    # Loading the network

    net = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # classes = []

    # Drawing the filtered bounding boxes with their label on the image

    with open(args.labels, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Geting the output layer from YOLO

    layers = net.getLayerNames()
    outputLayers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

    videoCapture = cv2.VideoCapture(0)

    while True:

        # Capturing the video

        ret, frame = videoCapture.read()

        img = cv2.resize(frame, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(outputLayers)

        classIds = []
        boundingBoxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                if confidence > confidenceThreshold:
                    centreX, centreY, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(centreX - w / 2)
                    y = int(centreY - h / 2)

        # Now here the green light is being turned on and the red light is being turned off because what the user wants to be detected has been detected
        #
        # GPIO.output(3, GPIO.HIGH)
        # GPIO.output(4, GPIO.LOW)

                    boundingBoxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    classIds.append(int(classId))

            # Running NMS for the bounding boxes. This is so that we can filter overlapping and low confident bounding boxes

        indices = cv2.dnn.NMSBoxes(boundingBoxes, confidences, confidenceThreshold, nmsThreshold)

        if len(indices) > 0:
            for index in indices:
                x, y, w, h = boundingBoxes[index]
                cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
                cv2.putText(img, classes[classIds[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, colors[index],
                            2)

        cv2.imshow("Image", cv2.resize(img, (800,600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()
