import cv2

color_image = cv2.imread("1.png")
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("1_after.png", gray_image)