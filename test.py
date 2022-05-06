# Imports

import cv2
import numpy as np
import argparse

# Code for testing the image

def testDetection():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", default = 'Test images\p1.jpeg', help = "Test image")

    # I added r before the file path because errors came without it and the original file path that wasnt the full one also gave errors.
    # The r basically converts a normal string to a raw string to prevent the error

    parser.add_argument("--cfg",
                        default = r'C:\Users\aliya\PycharmProjects\FinalProject\darknet\cfg\yolov4-tiny-custom.cfg',
                        help = "YOLO cfg file")
    parser.add_argument("--weights",
                        default = r'C:\Users\aliya\PycharmProjects\FinalProject\training\yolov4-tiny-custom_best.weights',
                        help = "YOLOV4-tiny weights file")
    parser.add_argument("--labels", default='darknet/data/obj.names', help="labels file")
    args = parser.parse_args()

    # Confidence threshold means it will only show the images that YOLO is more confident than the confident threshold humber (percentage)

    confidenceThreshold = 0

    # NMS one is for the non-max suppression so that we don't get more than one detection for one image

    nmsThreshold = 0.5

    # Loading the network

    net = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Geting the output layer from YOLO

    layers = net.getLayerNames()
    outputLayers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

    # Reading and converting the image to blob and performing the forward pass to get the bounding boxes with their confidence scores
    img = cv2.imread(args.image)
    height, width = img.shape[:2]

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

                boundingBoxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIds.append(int(classId))

    # Running NMS for the bounding boxes. This is so that we can filter overlapping and low confident bounding boxes

    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidences, confidenceThreshold, nmsThreshold).flatten().tolist()

    # Drawing the filtered bounding boxes with their label on the image

    with open(args.labels, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indices) > 0:
        for index in indices:
            x, y, w, h = boundingBoxes[index]
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
            cv2.putText(img, classes[classIds[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, colors[index],
                        2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()