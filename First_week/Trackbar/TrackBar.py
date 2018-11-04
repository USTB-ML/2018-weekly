import cv2
import numpy as np

pre_image = None
def on_Trackbar():
    global pre_image
    param = cv2.getTrackbarPos('threshold_circle', 'after')
    min_radium = cv2.getTrackbarPos('min_radium', 'after')
    circles = cv2.HoughCircles(img2, cv2.HOUGH_GRADIENT, 1, 0.1,
                               param1=150, param2=param, minRadius=min_radium, maxRadius=200)
    if not circles is None:
        circles = np.uint16(np.around(circles))
        img_copy = np.copy(img)

        for i in circles[0]:
            cv2.circle(img_copy, (i[0], i[1]), i[2], (0, 255, 0), 1)
            cv2.circle(img_copy, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow("after", img_copy)
        pre_image = img_copy
    
    else:
        cv2.imshow("after", pre_image)


# read the image
img = cv2.imread('D:/pictures/4.jpg')
cv2.imshow('before', img)

# pretreatment
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# define structure element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

img2 = cv2.medianBlur(closed, 5)

# creatTrackbar , use simple callback function
cv2.namedWindow('after', 1)
cv2.createTrackbar('threshold_circle', 'after', 30, 150, on_Trackbar)
cv2.createTrackbar('min_radium', 'after', 30,200, on_Trackbar)

while (1):
    on_Trackbar()
    cv2.waitKey(1)

cv2.destroyAllWindows() 
