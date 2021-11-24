# Imports

import cv2
import numpy as np#

# Imports for the led control

# RPi.GPIO allows us to control all GPIOs from the raspberry pi GPIO header

import RPi.GPIO as GPIO

# Time module allows us to turn the leds on for certain amounts of times

import time

# LED_PIN = 17

# This creates a constant global variable with the GPIO number for the LED

# GPIO.setmode(GPIO.BCM)

# https://roboticsbackend.com/raspberry-pi-control-led-python-3/

# # The 1 value is there because I want it to load and read a colour image
# CFB_img = cv2.imread('CaramelColourStages/CFB.jpg', 1)
# CBS_img = cv2.imread('CaramelColourStages/CBS.jpg', 1)
#
# CFB_hsv_img = cv2.cvtColor(CFB_img, cv2.COLOR_BGR2HSV)
# CBS_hsv_img = cv2.cvtColor(CBS_img, cv2.COLOR_BGR2HSV)
#
# print(CFB_hsv_img)
# print(CBS_hsv_img)
#
# # Create a VideoCapture object and read from input file
#
# capture = cv2.VideoCapture(0)
#
# # Checking if camera is opened successfully
#
# if (capture.isOpened() == False):
#     print("Error opening the video file")
#
# # Read until live video is over
#
# while (capture.isOpened()):
#
#     # Capture frame-by-frame
#
#     ret, frame = capture.read()
#
#     if ret == True:
#
#     # Display the resulting frame
#
#         # cv2.imshow('Frame', frame)
#
#     # Press Q on keyboard to quit
#
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#
#         # Convert the colour space from BGR to HSV
#
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#         # Pick a lower and upper bound for pixels we want to extact
#
#         # Check the hsv colour wheel to put inside the square brackets
#
#         lower_caramel = np.array([CFB_hsv_img])
#         upper_caramel = np.array([CBS_hsv_img])
#
#         # Mask = portion of an image - tells you which parts of the image you should keep
#         # The mask blocks out all the other colours other than the ones in the range of what youre wanting
#
#         mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
#         # This line is what picks out what colours match and what doesn't and turns it to black
#
#         result = cv2.bitwise_and(frame, frame, mask=mask)
#
#         cv2.imshow('frame', result)
#
#     # Break the loop
#     else:
#         break
#
# # Release the video capture object when everything is done
#
# capture.release()
#
# # Closes all the frames
#
# cv2.destroyAllWindows()

color = range of colours in acceptable form

if cameracolour is between lowercolour and lowestofcolour then
    yellow light

    else if cameracolour = colour
    greenlight

    else if cameracolour is between highest colour and burnt colour max then
    redlight

    else
    "Please point the camera at the caramel"

