import cv2

print("GeeksForGeeks")
print("Your OpenCV version is: " + cv2.__version__)

img = cv.imread

# Test again

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)