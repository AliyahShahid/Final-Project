import cv2

print("GeeksForGeeks")
print("Your OpenCV version is: " + cv2.__version__)

# The 1 value is there because I want it to load and read a colour image
img = cv2.imread('CaramelColourStages/CFB.jpg', 1)

CFB_hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(CFB_hsv_img)