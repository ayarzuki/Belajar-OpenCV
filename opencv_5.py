import cv2
import numpy as np
import math

# a = np.array([22]).astype(np.uint8)
# print(~a)

# print(cv2.bitwise_not(a))


# # A. Region Mask

# # Bitwise not untuk mask lingkaran putih

# img = cv2.imread("pertemuan11\\lena.jpg")
# h, w, c = img.shape

# mask = np.zeros((h, w)).astype(np.uint8)
# cv2.circle(mask, (h//2, w//2), 100, (255, 255, 255), -1)

# mask_inv = cv2.bitwise_not(img, mask=mask)

# cv2.imshow('original', img)
# cv2.imshow('mask', mask)
# cv2.imshow('mask_inv', mask_inv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Bitwise Not untuk mask hasil thresholding
# img = cv2.imread("pertemuan11\\hand.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
# mask_inv = cv2.bitwise_not(img, mask=thresh)

# cv2.imshow('Bitwise Not', mask_inv)
# cv2.imshow('Mask', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Menambahkan trackbar untuk memvariasikan nilai threshold sebelum di apply Bitwise Not
# max_value = 255
# default_value = 120

# title_window = 'Bitwise Not'


# def on_trackbar(val):
#     ret, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY_INV)
#     mask_inv = cv2.bitwise_not(img, mask=thresh)
#     cv2.imshow(title_window, mask_inv)


# img = cv2.imread("pertemuan11\\hand.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('thresh', title_window,
#                    default_value, max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # B. Bitwise AND

# a = np.array([100]).astype(np.uint8)
# b = np.array([100]).astype(np.uint8)

# print(a & b)
# print(cv2.bitwise_and(a, b))

# # Bitwise AND untuk mask kotak putih
# img1 = cv2.imread("pertemuan11\\lena.jpg")
# img2 = cv2.imread("pertemuan11\\apple.jpg")
# h, w, c = img1.shape

# mask = np.zeros((h, w)).astype(np.uint8)
# cv2.rectangle(mask, (h//4, w//4), (3*h//4, 3*w//4), (255, 255, 255), -1)

# mask_and = cv2.bitwise_and(img2, img2, mask=mask)

# cv2.imshow('mask_inv', mask_and)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Bitwise AND untuk mask hasil thresholding
# img1 = cv2.imread("pertemuan11\\apple.jpg")
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# img2 = cv2.imread("pertemuan11\\lena.jpg")
# mask_inv = cv2.bitwise_and(img1, img2, mask=thresh)

# cv2.imshow('Bitwise Or', mask_inv)
# cv2.imshow('Apple mask', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Bitwise OR

# a = np.array([100]).astype(np.uint8)
# b = np.array([40]).astype(np.uint8)
# print(a | b)
# print(cv2.bitwise_or(a, b))

# # A. Bitwise OR untuk mask lingkaran putih
# img1 = cv2.imread("pertemuan11\\lena.jpg")
# img2 = cv2.imread("pertemuan11\\apple.jpg")
# h, w, c = img1.shape

# mask = np.zeros((h, w)).astype(np.uint8)
# cv2.circle(mask, (h//2, w//2), 180, (255, 255,  255), -1)

# mask_and = cv2.bitwise_or(img1, img2, mask=mask)

# cv2.imshow('mask_inv', mask_and)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # B. Bitwise Or untuk mask hasil thresholding
# img1 = cv2.imread("pertemuan11\\apple.jpg")
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# img2 = cv2.imread("pertemuan11\\lena.jpg")
# mask_inv = cv2.bitwise_or(img2, img2, mask=thresh)

# cv2.imshow('Bitwise Or', mask_inv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 2. Range Thresholding

# # A. Convert RGB value to HSV (cv2.cvtColor())
# blue = np.uint8([[[255, 0, 0]]])
# hsv_green = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
# print(hsv_green)

# # B. Detect Red, Green and Blue Color from image

# img = cv2.imread('pertemuan11\\blocks.jpg')
# # convert to hsv
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # define range of blue color in HSV
# lower_blue = np.array([110, 50, 50])
# upper_blue = np.array([130, 255, 255])

# # define range of red color in HSV
# lower_red = np.array([-10, 50, 50])
# upper_red = np.array([10, 255, 255])

# # define range of green color in HSV
# lower_green = np.array([50, 50, 50])
# upper_green = np.array([70, 255, 255])

# # Threshold the HSV image to get only blue colors
# mask_blue = cv2.inRange(hsv.copy(), lower_blue, upper_blue)
# mask_red = cv2.inRange(hsv.copy(), lower_red, upper_red)
# mask_green = cv2.inRange(hsv.copy(), lower_green, upper_green)

# mask = mask_blue + mask_red + mask_green

# res = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow('frame', img)
# cv2.imshow('res', res)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Define range of yellow color in HSV
# lower_yellow = np.array([25, 50, 50])
# upper_yellow = np.array([35, 255, 255])

# cap = cv2.VideoCapture("pertemuan11\\yellow_ball.mp4")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv.copy(), lower_yellow, upper_yellow)
#         res = cv2.bitwise_and(frame, frame, mask=mask)
#         cv2.imshow('Detected Object', res)

#         if cv2.waitKey(100) == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

# # 2. Shape Detection

# # A. Hough Line Transform (Line Detector)
# img = cv2.imread("pertemuan11\\road.jpg")

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 200)

# lines = cv2.HoughLines(edges, 1, np.pi/180, 100, None, 0, 0)

# for line in lines:
#     rho = line[0][0]
#     theta = line[0][1]
#     a = math.cos(theta)
#     b = math.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1, y1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#     x2, y2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)

# cv2.imshow("Hough Line Transform", img)
# cv2.imshow("Gray", gray)
# cv2.imshow("Edge", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(rho)
# print(theta)
# print(a)
# print(b)
# print(x0)
# print(y0)

# # B. Trackbar mengatur nilai threshold Hough Line Transform (Line Detector)
# max_value = 300
# default_value = 150

# title_window = "Hough Line Transform"


# def on_trackbar(val):
#     frame = img.copy()
#     lines = cv2.HoughLines(edges, 1, np.pi/180, val, None, 0, 0)
#     if lines is not None:
#         for line in lines:
#             rho = line[0][0]
#             theta = line[0][1]
#             a = math.cos(theta)
#             b = math.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1, y1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#             x2, y2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#             cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)

#     cv2.imshow(title_window, frame)


# img = cv2.imread('pertemuan11\\road.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 200)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('thresh', title_window,
#                    default_value, max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # C. Probabilistic Hough Transform cv2.HoughLinesP()

# img = cv2.imread('pertemuan11\\road.jpg')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 200)
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
#                         minLineLength=50, maxLineGap=30)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.imshow("Result Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # D. Trackbar untuk mengatur minLineLength dan maxLineGap
# maxLineGap = 20
# minLineLength = 100

# title_window = "Hough Line Transform"


# def draw_line(lines):
#     frame = img.copy()
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)

#     cv2.imshow(title_window, frame)


# def on_trackbar_minLineLength(val):
#     global minLineLength
#     minLineLength = val
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
#                             minLineLength=minLineLength, maxLineGap=maxLineGap)
#     draw_line(lines)


# def on_trackbar_maxLineGap(val):
#     global maxLineGap
#     maxLineGap = val
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
#                             minLineLength=minLineLength, maxLineGap=maxLineGap)
#     draw_line(lines)


# img = cv2.imread('pertemuan11\\road.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 200)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('maxGap', title_window, 20, 200, on_trackbar_maxLineGap)
# cv2.createTrackbar('minLen', title_window, 100, 200, on_trackbar_minLineLength)

# on_trackbar_minLineLength(100)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 3. Hough Circle Transform
# # A. Hough Circle Transform using Trackbar
# img = cv2.imread("pertemuan11\\eye.jpg")
# h, w, c = img.shape

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur = cv2.GaussianBlur(gray, (5, 5), 0, 0)

# circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1,
#                            h/64, param1=200, param2=17, minRadius=21, maxRadius=30)

# if circles is not None:
#     circles = np.uint16(np.around(circles))[0]
#     for i in circles:
#         cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)


# cv2.imshow("output", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # B. Hough Circle Transform using Trackbar
# minRadius = 21
# maxRadius = 30
# param1 = 200
# param2 = 17

# title_window = "Hough Circle Transform"


# def draw_circle(circles):
#     frame = img.copy()
#     if circles is not None:
#         circles = np.uint16(np.around(circles))[0]
#         for i in circles:
#             cv2.circle(frame, (i[0], i[1]), i[2],
#                        (0, 255, 255), 2, cv2.LINE_AA)

#     cv2.imshow(title_window, frame)


# def on_trackbar_minRadius(val):
#     global minRadius
#     minRadius = val
#     circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, h/64, param1=param1,
#                                param2=param2, minRadius=minRadius, maxRadius=maxRadius)
#     draw_circle(circles)


# def on_trackbar_maxRadius(val):
#     global maxRadius
#     maxRadius = val
#     circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, h/64, param1=param1,
#                                param2=param2, minRadius=minRadius, maxRadius=maxRadius)
#     draw_circle(circles)


# def on_trackbar_param1(val):
#     global param1
#     param1 = val
#     circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, h/64, param1=param1,
#                                param2=param2, minRadius=minRadius, maxRadius=maxRadius)
#     draw_circle(circles)


# def on_trackbar_param2(val):
#     global param2
#     param2 = val
#     circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, h/64, param1=param1,
#                                param2=param2, minRadius=minRadius, maxRadius=maxRadius)
#     draw_circle(circles)


# img = cv2.imread('pertemuan11\\eye.jpg')
# h, w, c = img.shape
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 0, 0)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('minRadius', title_window, 21, 50, on_trackbar_minRadius)
# cv2.createTrackbar('maxRadius', title_window, 30, 50, on_trackbar_maxRadius)
# cv2.createTrackbar('param1', title_window, 200, 255, on_trackbar_param1)
# cv2.createTrackbar('param2', title_window, 17, 30, on_trackbar_param2)

# on_trackbar_minRadius(minRadius)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # TaskDectect Yellow Ball on video yellow_bal.mp4 using Hough Circle
# cap = cv2.VideoCapture("pertemuan11\\yellow_ball.mp4")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         circles = cv2.HoughCircles(
#             gray, cv2.HOUGH_GRADIENT, 1, 40, param1=180, param2=17, minRadius=21, maxRadius=100)

#         if circles is not None:
#             circles = np.uint16(np.around(circles))[0]
#             for i in circles:
#                 cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

#         cv2.imshow('Hough Circle - Video', frame)

#         if cv2.waitKey(10) == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

# # Application Hough Lines
# # Real-time Road Line Detection
# maxLineGap = 200
# minLineLength = 50
# title_window = "Road Lane Detector"
# poly = [[59, 339], [293, 217], [434, 210], [613, 325]]
# is_draw = False


# def on_trackbar_minLineLength(val):
#     global minLineLength
#     minLineLength = val


# def on_trackbar_maxLineGap(val):
#     global maxLineGap
#     maxLineGap = val


# def read_poly(event, x, y, flags, param):
#     global poly, is_draw

#     if event == cv2.EVENT_RBUTTONDOWN:
#         is_draw = True
#         poly = []

#     if event == cv2.EVENT_LBUTTONDOWN and is_draw:
#         poly.append([x, y])

#     if len(poly) == 4 and is_draw:
#         is_draw = False


# cv2.namedWindow(title_window)
# cv2.createTrackbar('maxGap', title_window, 200, 400, on_trackbar_maxLineGap)
# cv2.createTrackbar('minLen', title_window, 50, 400, on_trackbar_minLineLength)
# cv2.setMouseCallback(title_window, read_poly)

# cap = cv2.VideoCapture('pertemuan11\drive.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         if not is_draw:
#             stencil = np.zeros_like(gray)
#             polygon = np.array(poly)
#             cv2.fillConvexPoly(stencil, polygon, 1)
#             roi = cv2.bitwise_and(gray, gray, mask=stencil)

#             ret, thresh = cv2.threshold(roi, 130, 255, cv2.THRESH_BINARY)
#             cv2.imshow('roi', roi)

#             lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=20,
#                                     minLineLength=minLineLength, maxLineGap=maxLineGap)
#             if lines is not None:
#                 for line in lines:
#                     x1, y1, x2, y2 = line[0]
#                     cv2.line(frame, (x1, y1), (x2, y2),
#                              (255, 0, 255), 1, cv2.LINE_AA)

#         cv2.imshow(title_window, frame)

#         if cv2.waitKey(25) == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

# # Road Line Detection

title_window = "Road Lane Detector"
roi_poly = np.array([[59, 339], [293, 217], [434, 210], [613, 325]])

cap = cv2.VideoCapture('pertemuan11\\drive.mp4')

r_m, l_m, r_c, l_c = [], [], [], []


def draw_lines(shape, lines, thickness=3, scale=0.65):
    global r_m, l_m, r_c, l_c
    h, w = shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y1-y2)/(x1-x2)
                if slope > 0.3:
                    yintercept = y2 - (slope*x2)
                    r_m.append(slope)
                    r_c.append(yintercept)
                elif slope < -0.3:
                    yintercept = y2 - (slope*x2)
                    l_m.append(slope)
                    l_c.append(yintercept)

    avg_l_m = np.mean(l_m[-30:])
    avg_l_c = np.mean(l_c[-30:])
    avg_r_m = np.mean(r_m[-30:])
    avg_r_c = np.mean(r_c[-30:])

    try:
        y1, y2 = int(scale*h), h
        l_x1 = int((y1 - avg_l_c)/avg_l_m)
        l_x2 = int((y2 - avg_l_c)/avg_l_m)
        r_x1 = int((y1 - avg_r_c)/avg_r_m)
        r_x2 = int((y2 - avg_r_c)/avg_r_m)

        pts = np.array([[l_x1, y1], [l_x2, y2], [r_x2, y2],
                        [r_x1, y1]]).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (0, 127, 50))
        cv2.line(img, (l_x1, y1), (l_x2, y2), (0, 255, 255), thickness)
        cv2.line(img, (r_x1, y1), (r_x2, y2), (0, 255, 255), thickness)
        return img
    except ValueError:
        pass


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = np.zeros(gray.shape).astype(np.uint8)
        cv2.fillPoly(mask, [roi_poly], (255, 255, 255))
        roi = cv2.bitwise_and(gray, gray, mask=mask)

        ret, thresh = cv2.threshold(roi, 130, 255, cv2.THRESH_BINARY)
        edged = cv2.Canny(thresh, 127, 200)

        lines = cv2.HoughLinesP(edged, 1, np.pi/180, 10,
                                np.array([]), minLineLength=20, maxLineGap=5)

        overlay = draw_lines(edged.shape, lines, thickness=3, scale=0.65)
        frame = cv2.addWeighted(overlay, 1, frame, 0.8, 0)

        cv2.imshow(title_window, frame)
        cv2.imshow(title_window + "- ROI", roi)
        cv2.imshow(title_window + "- Edge", edged)
        cv2.imshow(title_window + "- thresh", thresh)
        if cv2.waitKey(0) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
