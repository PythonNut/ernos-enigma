import cv2
import numpy as np
import scipy.spatial
from skimage import img_as_ubyte
import collections
import copy

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

def query_ball_midpoint(kd, c1, c2):
    return kd.query_ball_point((c1 + c2)/2, 10)

def angle_relative_to(center, mapping):
    def angle_closure(index):
        x, y = mapping[index] - mapping[center]
        angle = -np.arctan2(y, x)
        if angle < 0:
            angle += 2 * np.pi
        return angle
    return angle_closure

def rotate(l, n):
    return l[n:] + l[:n]

while True:
    ret_val, img = cam.read()
    cv2.imshow("input", img)
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
    cv2.imshow("connected regions", s*255)
    # s *= 255
    # s = iterative_refine(s, 1)
    # s2 = cv2.threshold(s, 127, 255, cv2.THRESH_BINARY)[1]
    connected = cv2.connectedComponentsWithStats(s, 8, cv2.CV_32S)

    # lsd = cv2.createLineSegmentDetector(0)
    # lines = lsd.detect(s)[0]
    # lsd.drawSegments(s,lines)

    s2 = np.zeros(s.shape)
    s3 = np.zeros(s.shape + (3,), np.uint8)
    num_labels, labels, stats, centroids = connected
    interesting_centroids = []
    interesting_centroid_labels = []
    for label in range(num_labels):
        centroid = centroids[label]
        area = stats[label, cv2.CC_STAT_AREA]
        base_size = 500
        max_multiplier = 10
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
        interesting_centroid_labels.append(label)

        # contours,hierarchy,_ = cv2.findContours(as_u8(labels == label), 1, 2)
        # print(contours)
        # cv2.drawContours(s2, contours, -1, (0,255,0), 3)
        # print(area)

    cv2.imshow("patches", s2)

    # print(interesting_centroids)
    corner_count = [0 for _ in range(len(interesting_centroids))]
    # corner_count = np.zeros(len(interesting_centroids))

    # print(np.vstack(interesting_centroids))
    if not interesting_centroids:
        print("No interesting centroids!")
        continue

    kd = scipy.spatial.cKDTree(np.vstack(interesting_centroids))
    for i, c1 in enumerate(interesting_centroids):
        for j, c2 in enumerate(interesting_centroids):
            if j >= i: continue
            if len(query_ball_midpoint(kd, c1, c2)) == 1:
                corner_count[i] += 1
                corner_count[j] += 1

    # print(corner_count)

    corners = []
    for i in range(len(interesting_centroids)):
        if corner_count[i] == 3:
            corners.append(i)

    if len(corners) != 4:
        print("Need exactly four corners!")

    min_corner_dist = 1e8
    max_corner_dist = 1e-8
    for i in corners:
        for j in corners:
            if j >= i: continue
            c1, c2 = interesting_centroids[i], interesting_centroids[j]
            dist = np.linalg.norm(c1 - c2)
            if dist > max_corner_dist:
                max_corner_dist = dist
            if dist < min_corner_dist:
                min_corner_dist = dist
    # print(max_corner_dist, min_corner_dist)

    if max_corner_dist/min_corner_dist > 2:
        print("Corner distances don't make sense!")

    for i, centroid in enumerate(interesting_centroids):
        color = (255, 255, 255)
        if corner_count[i] == 3:
            color = (255, 0, 255)
        # if corner_count[i] == 1:
        #     color = (255, 0, 0)
        cv2.circle(s3, (int(centroid[0]), int(centroid[1])), 10, color, 4)

    nine_points = set(corners)
    center = None
    for i in corners:
        for j in corners:
            if j >= i: continue
            c1, c2 = interesting_centroids[i], interesting_centroids[j]
            midpoints = query_ball_midpoint(kd, c1, c2)
            if len(midpoints) != 1: continue
            midpoint_idx = midpoints[0]
            if midpoint_idx not in nine_points:
                nine_points.add(midpoint_idx)
            else:
                center = midpoint_idx

    if len(nine_points) != 9:
        print(f"9 point set has {len(nine_points)} points!")

    if center is None:
        print("Failed to find center of face!")
    else:
        eight_points = list(nine_points - set([center]))
        # print(interesting_centroids[center], [interesting_centroids[p] for p in  eight_points])
        print([angle_relative_to(center, interesting_centroids)(e) for e in eight_points])
        eight_points.sort(key=angle_relative_to(center, interesting_centroids))

        if eight_points[0] not in corners:
            eight_points = rotate(eight_points, 1)

        print(eight_points)

        if len(eight_points) == 8:
            first_point = interesting_centroids[eight_points[0]]
            cv2.circle(s3, (int(first_point[0]), int(first_point[1])), 3, (0,0,255), 4)

            first_point = interesting_centroids[eight_points[1]]
            cv2.circle(s3, (int(first_point[0]), int(first_point[1])), 3, (0,255,255), 4)

            first_point = interesting_centroids[center]
            cv2.circle(s3, (int(first_point[0]), int(first_point[1])), 3, (255,0,0), 4)

    # s /= np.ndarray(s).max()
    # cv2.normalize(s,  s2, 0, 255, cv2.NORM_MINMAX)
    # s2 = s.convertTo(cs2.CV_8UC1)
    cv2.imshow("output", s3)
    # cv2.imshow("output", (connected[1] > 0).astype('uint8')*255)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
