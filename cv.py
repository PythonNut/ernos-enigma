import cv2
import numpy as np

cam = cv2.VideoCapture(2)

while True:
    ret_val, img = cam.read()

    cv2.imshow("output", img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
