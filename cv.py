import cv2
import numpy as np
from skimage import img_as_ubyte
# from pylsd.lsd import lsd

cam = cv2.VideoCapture(2)

def iterative_refine(img, iterations=1):
    kernel = np.ones((3,3),np.uint8)
    for _ in range(2):
        img = cv2.erode(img,kernel,iterations=iterations)
        img = cv2.dilate(img,kernel,iterations=iterations)
    return img

def as_u8(mat):
    return mat.astype('uint8')

while True:
    ret_val, img = cam.read()
    img = cv2.GaussianBlur(img,(5,5),0)
    # lap = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    s = sobelx**2 + sobely**2
    s = s/np.max(s) * 255
    s = s.astype('uint8')
    s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    s = s > 1
    s = s.astype('uint8')
    s = 1 - s
    # s *= 255
    # s = iterative_refine(s, 1)
    # s2 = cv2.threshold(s, 127, 255, cv2.THRESH_BINARY)[1]
    connected = cv2.connectedComponentsWithStats(s, 8, cv2.CV_32S)

    # lsd = cv2.createLineSegmentDetector(0)
    # lines = lsd.detect(s)[0]
    # lsd.drawSegments(s,lines)

    s2 = np.zeros(s.shape)
    s3 = np.zeros(s.shape)
    num_labels, labels, stats, centroids = connected
    interesting_centroids = []
    for label in range(num_labels):
        centroid = centroids[label]
        area = stats[label, cv2.CC_STAT_AREA]
        base_size = 500
        max_multiplier = 6
        w, h = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        max_area = w * h
        # print(cv2.arcLength(as_u8(labels == label).astype('us'), True))
        if w > 200 or h > 200:
            continue
        if max_area > area * 2:
            continue
        if not (area > base_size and area < base_size * max_multiplier):
            continue

        s2 += labels == label
        interesting_centroids.append(centroid)

        # contours,hierarchy,_ = cv2.findContours(as_u8(labels == label), 1, 2)
        # print(contours)
        # cv2.drawContours(s2, contours, -1, (0,255,0), 3)
        # print(area)

    print(interesting_centroids)
    for centroid in interesting_centroids:
        cv2.circle(s3, (int(centroid[0]), int(centroid[1])), 10, (255, 255, 255), 4)



    # for l in lines:
    #     x0, y0, x1, y1 = l.flatten()
    #     length = np.linalg.norm([x0 - x1, y0 - y1])
    #     if length < 10:
    #         continue
    #     if length > 100:
    #         continue
    #     # print(length)
    #     cv2.line(s2, (x0, y0), (x1,y1), 255, 1, cv2.LINE_AA)
    # lines = cv2.HoughLinesP(s,1,np.pi/45,100,10,5)

    # # print(lines)
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(s2, (x1,y1),(x2,y2),(255,255,255),2)

    # s /= np.ndarray(s).max()
    # cv2.normalize(s,  s2, 0, 255, cv2.NORM_MINMAX)
    # s2 = s.convertTo(cs2.CV_8UC1)
    cv2.imshow("output", s3)
    # cv2.imshow("output", (connected[1] > 0).astype('uint8')*255)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
