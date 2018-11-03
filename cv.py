import cv2
import numpy as np
import scipy.spatial
from skimage import img_as_ubyte
import collections
import copy
import time
import shelve
import socket
import traceback
from pathlib import Path

CONNECT_TO_SERVER = False

cam = cv2.VideoCapture(0)

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

def cvtPixel(color, conversion):
    return cv2.cvtColor(np.array([[color]]).astype('uint8'), conversion)[0,0]

def hue_delta(h1, h2):
    d = abs(h1- h2)
    return min(d, 180 - d)

def hsv_metric(c1, c2):
    h1, s1, v1 = c1.astype('float')
    h2, s2, v2 = c2.astype('float')
    return (hue_delta(h1, h2))**2 + (abs(s1 - s2)/2)**2 + (abs(v1 - v2)/8)**2

CALIBRATION_ORDER = "RWBYGO"
CALIBRATION_INDEX = 0
CALIBRATION_READY = False
CALIBRATION_START_TIME = 0

COLOR_MAP = {}
DISPLAY_COLOR_MAP = {
    "X": (0,0,0),
    "R": (0,0,255),
    "W": (255,255,255),
    "B": (255,0,0),
    "Y": (0,255,255),
    "G": (0,255,0),
    "O": (0,165,255)
}

if Path('calibration.dat.dat').exists() or Path('calibration.dat').exists():
    with shelve.open('calibration.dat') as s:
        COLOR_MAP = s['COLOR_MAP']
        CALIBRATION_INDEX = 6


def guess_color(c):
    min_dist = float('inf')
    min_name = "X"
    for name, color in COLOR_MAP.items():
        dist = hsv_metric(c, color)
        if dist < min_dist:
            min_dist = dist
            min_name = name

    return min_name

def stochastic_pixel_sampler(img, x, y, radius=10, samples=100):
    accumulator = np.zeros(3)
    for _ in range(samples):
        dx, dy = float('inf'), float('inf')
        while dx**2 + dy**2 > radius**2:
            dx = np.random.rand() * radius
            dy = np.random.rand() * radius
        accumulator += img[int(y + dy), int(x + dx)]
    return accumulator / samples

def draw_circle(img, p, r, c, t = 4):
    cv2.circle(img, (int(p[0]), int(p[1])), r, c, t)

def set_up_client_socket(host,port):
    #please close the socket later
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    return s

def send_data(socket, data):
    socket.sendall(bytes(data, 'utf-8'))

try:
    if CONNECT_TO_SERVER:
        server = set_up_client_socket('localhost', 9998)
    while True:
        try:
            # print(COLOR_MAP)
            if CALIBRATION_INDEX < len(CALIBRATION_ORDER) and not CALIBRATION_READY:
                side = CALIBRATION_ORDER[CALIBRATION_INDEX]
                input("Please face side {} > ".format(side))
                CALIBRATION_READY = True
                CALIBRATION_START_TIME = time.process_time()

            ret_val, img = cam.read()
            img = cv2.GaussianBlur(img,(5,5),0)
            cv2.imshow("input", img)
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
                if w > 100 or h > 100:
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
                draw_circle(s3, centroid, 10, color, 4)

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

            s4 = np.zeros(s.shape + (3,), np.uint8)
            if len(nine_points) != 9:
                print("9 point set has {} points!".format(len(nine_points)))

            if center is None:
                print("Failed to find center of face!")
            else:
                eight_points = list(nine_points - set([center]))
                # print(interesting_centroids[center], [interesting_centroids[p] for p in  eight_points])
                # print([angle_relative_to(center, interesting_centroids)(e) for e in eight_points])
                eight_points.sort(key=angle_relative_to(center, interesting_centroids))

                if eight_points[0] not in corners:
                    eight_points = rotate(eight_points, 1)

                # print(eight_points)

                if len(eight_points) == 8:
                    first_point = interesting_centroids[eight_points[0]]
                    draw_circle(s3, first_point, 3, (0,0,255), 4)

                    first_point = interesting_centroids[eight_points[1]]
                    draw_circle(s3, first_point, 3, (0,255,255), 4)

                    first_point = interesting_centroids[center]
                    draw_circle(s3, first_point, 3, (255,0,0), 4)

                    msg = ""
                    rgb_color = stochastic_pixel_sampler(img, first_point[0], first_point[1])
                    hsv_color = cvtPixel(rgb_color, cv2.COLOR_BGR2HSV)
                    color_name = guess_color(hsv_color)
                    draw_circle(s4, first_point, 16, DISPLAY_COLOR_MAP[color_name], 4)
                    print(color_name, end="")
                    msg += color_name

                    for i in eight_points:
                        point = interesting_centroids[i]
                        rgb_color = stochastic_pixel_sampler(img, point[0], point[1])
                        hsv_color = cvtPixel(rgb_color, cv2.COLOR_BGR2HSV)
                        color_name = guess_color(hsv_color)

                        draw_circle(s4, point, 16, DISPLAY_COLOR_MAP[color_name], 4)

                        msg += color_name
                        # print(hsv_color)
                        print(color_name, end="")
                    print()
                    if CONNECT_TO_SERVER:
                        server.sendall(bytes(msg, 'utf-8'))

                    # Wow, we did it
                    if CALIBRATION_READY and time.process_time() - CALIBRATION_START_TIME > 1:
                        color = CALIBRATION_ORDER[CALIBRATION_INDEX]
                        color_acc = np.zeros(3)
                        for i in nine_points:
                            point = interesting_centroids[i]
                            color_acc += stochastic_pixel_sampler(img, point[0], point[1])
                            # color_acc += img[int(point[0]), int(point[1])]

                        color_acc /= 9
                        hsv_color = cvtPixel(color_acc, cv2.COLOR_BGR2HSV)

                        COLOR_MAP[color] = hsv_color

                        print("gottem")

                        CALIBRATION_INDEX += 1
                        CALIBRATION_READY = False

                        if CALIBRATION_INDEX == len(CALIBRATION_ORDER):
                            with shelve.open('calibration.dat') as s:
                                s['COLOR_MAP'] = COLOR_MAP

            # s /= np.ndarray(s).max()
            # cv2.normalize(s,  s2, 0, 255, cv2.NORM_MINMAX)
            # s2 = s.convertTo(cs2.CV_8UC1)
            cv2.imshow("output", s3)
            cv2.imshow("colors", s4)
            # cv2.imshow("output", (connected[1] > 0).astype('uint8')*255)
            if cv2.waitKey(1) == 27:
                break
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exec()

    cv2.destroyAllWindows()

finally:
    if CONNECT_TO_SERVER:
        server.close()
