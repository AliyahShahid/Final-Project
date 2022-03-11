# # Imports
#
# import cv2
# # import numpy as np
# import matplotlib.pyplot as plt
#
# # The 1 value is there because I want it to load and read a colour image
# CFB_img = cv2.imread('CaramelColourStages/CFB.jpg', 1)
# CCL_img = cv2.imread('CaramelColourStages/CCL.jpg', 1)
# CBL_img = cv2.imread('CaramelColourStages/CBL.jpg', 1)
# CLA_img = cv2.imread('CaramelColourStages/CLA.jpg', 1)
# CMA_img = cv2.imread('CaramelColourStages/CMA.jpg', 1)
# CDA_img = cv2.imread('CaramelColourStages/CDA.jpg', 1)
# CBS_img = cv2.imread('CaramelColourStages/CBS.jpg', 1)
#
# CFB_hsv_img = cv2.cvtColor(CFB_img, cv2.COLOR_BGR2HSV)
# CCL_hsv_img = cv2.cvtColor(CCL_img, cv2.COLOR_BGR2HSV)
# CBL_hsv_img = cv2.cvtColor(CBL_img, cv2.COLOR_BGR2HSV)
# CLA_hsv_img = cv2.cvtColor(CLA_img, cv2.COLOR_BGR2HSV)
# CMA_hsv_img = cv2.cvtColor(CMA_img, cv2.COLOR_BGR2HSV)
# CDA_hsv_img = cv2.cvtColor(CDA_img, cv2.COLOR_BGR2HSV)
# CBS_hsv_img = cv2.cvtColor(CBS_img, cv2.COLOR_BGR2HSV)
#
# # Loading the sugar image to use to find matching features and detect the sugar in the video
# # Converting the image here and video later into grayscale so that less information needs to be provided for each pixel
#
# sugar_img = cv2.imread('CaramelColourStages/CLA.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Create a VideoCapture object and read from input file
#
# capture = cv2.VideoCapture('CaramelColourStages/IMG_0091.MOV')
#
# # Checking if camera is opened successfully
#
# if (capture.isOpened() == False):
#     print("Error opening the video stream")
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
#         gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#
#         cv2.imshow('Frame', gray_video)
#
#     # Press Q on keyboard to quit
#     # Putting waitKey as more than 0 because I don't want to pause the stream forever
#
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#
#     # Break the loop
#     else:
#         break
#
# # Release the video capture object when everything is done
#
# capture.release()
#
# # # Initiating ORB detector
# #
# orb = cv2.ORB_create()
# #
# # # Finding the keypoints and descriptors with ORB
# #
# kp1, des1 = orb.detectAndCompute(sugar_img, None)
# kp2, des2 = orb.detectAndCompute(gray_video, None)
#
# # # Creating the BFMatcher object
# #
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# #
# # # Matching the descriptors
# #
# matches = bf.match(des1, des2)
# #
# # # Sorting them in the order of their distance
# #
# matches = sorted(matches, key = lambda x:x.distance)
#
# # Drawing the first 20 matches
#
# matched_img = cv2.drawMatches(sugar_img, kp1, gray_video, kp2, matches[:20], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
# # # Convert the colour space from Grayscale to HSV
#
# bgr_video = cv2.cvtColor(gray_video, cv2.COLOR_GRAY2BGR)
# hsv_video = cv2.cvtColor(bgr_video, cv2.COLOR_BGR2HSV)
#
# print(hsv_video)
#
# plt.imshow(matched_img), plt.show()
#
# cv2.waitKey(0)
#
# # Closes all the frames
#
# cv2.destroyAllWindows()
#
# # Unfinished stuff
#
# # color = range of colours in acceptable form
#
# # if cameracolour is between lowercolour and lowestofcolour then
# #     yellow light
#
# #     else if cameracolour = colour
# #     greenlight
#
# #     else if cameracolour is between highest colour and burnt colour max then
# #     redlight
#
# #     else
# #     "Please point the camera at the caramel"