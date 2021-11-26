# Imports

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading the sugar image to use to find matching features and detect the sugar in the video
# Converting the image here and video later into grayscale so that less information needs to be provided for each pixel

sugar_img = cv2.imread('CaramelColourStages/Sugar.jpg', cv2.IMREAD_GRAYSCALE)

# Create a VideoCapture object and read from input file

capture = cv2.VideoCapture('CaramelColourStages/Sugar_vid.MP4')

# Checking if camera is opened successfully

if (capture.isOpened() == False):
    print("Error opening the video stream")

# Read until live video is over

while (capture.isOpened()):

    # Capture frame-by-frame

    ret, frame = capture.read()

    if ret == True:

        gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame

        cv2.imshow('Frame', gray_video)

    # Press Q on keyboard to quit
    # Putting waitKey as more than 0 because I don't want to pause the stream forever

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# Release the video capture object when everything is done

capture.release()

# # Initiating ORB detector
#
orb = cv2.ORB_create()
#
# # Finding the keypoints and descriptors with ORB
#
kp1, des1 = orb.detectAndCompute(sugar_img, None)
kp2, des2 = orb.detectAndCompute(gray_video, None)

# # Creating the BFMatcher object
#
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Matching the descriptors
#
matches = bf.match(des1, des2)
#
# # Sorting them in the order of their distance
#
matches = sorted(matches, key = lambda x:x.distance)

# Drawing the first 20 matches

matched_img = cv2.drawMatches(sugar_img, kp1, gray_video, kp2, matches[:20], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(matched_img), plt.show()

cv2.waitKey(0)

# Closes all the frames

cv2.destroyAllWindows()

# # Convert the colour space from BGR to HSV
#
# hsv = cv2.cv2tColor(frame, cv2.COLOR_BGR2HSV)
#
# # Pick a lower and upper bound for pixels we want to extract
#
# # Check the hsv colour wheel to put inside the square brackets
#
# lower_caramel = np.array([CFB_hsv_img])
# upper_caramel = np.array([CBS_hsv_img])
#
# # Mask = portion of an image - tells you which parts of the image you should keep
# # The mask blocks out all the other colours other than the ones in the range of what youre wanting
#
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
# # This line is what picks out what colours match and what doesn't and turns it to black
#
# result = cv2.bitwise_and(frame, frame, mask=mask)
#
# cv2.imshow('frame', result)

# Unfinished stuff

# color = range of colours in acceptable form
#
# if cameracolour is between lowercolour and lowestofcolour then
#     yellow light
#
#     else if cameracolour = colour
#     greenlight
#
#     else if cameracolour is between highest colour and burnt colour max then
#     redlight
#
#     else
#     "Please point the camera at the caramel"

# Imports for the led control

# RPi.GPIO allows us to control all GPIOs from the raspberry pi GPIO header

# import RPi.GPIO as GPIO

# Time module allows us to turn the leds on for certain amounts of times

# import time

# LED_PIN = 17

# This creates a constant global variable with the GPIO number for the LED

# GPIO.setmode(GPIO.BCM)

# https://roboticsbackend.com/raspberry-pi-control-led-python-3/

# # The 1 value is there because I want it to load and read a colour image
# CFB_img = cv2.imread('CaramelColourStages/CFB.jpg', 1)
# CBS_img = cv2.imread('CaramelColourStages/CBS.jpg', 1)
#
# CFB_hsv_img = cv2.cv2tColor(CFB_img, cv2.COLOR_BGR2HSV)
# CBS_hsv_img = cv2.cv2tColor(CBS_img, cv2.COLOR_BGR2HSV)
#
# print(CFB_hsv_img)
# print(CBS_hsv_img)

# Initialising the ORB detector algorithm

    # orb = cv2.ORB_create()
    #
    # sugar_img_keyPoints, sugar_img_descriptors = orb.detectAndCompute(sugar_img_gray, None)
    # sugar_vid_keyPoints, sugar_vid_descriptors = orb.detectAndCompute(sugar_vid_gray, None)
    #
    # matcher = cv2.BFMatcher()
    # matches = matcher.match(sugar_vid_descriptors, sugar_img_descriptors)
    #
    # final_sugar_img = cv2.drawMatches(sugar_vid, sugar_vid_keyPoints, sugar_img, sugar_img_keyPoints, matches[:20], None)
    #
    # final_sugar_img = cv2.resize(final_sugar_img, (1000, 650))
    #
    # cv2.imshow("Matches", final_sugar_img)
    # cv2.waitKey(3000)

    # Find image in video
    # Put a box around the image
    # Label the image of the current stage
    # Track the image
