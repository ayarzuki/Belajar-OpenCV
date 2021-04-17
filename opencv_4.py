# import math
import cv2
import numpy as np

# # 1. OpenCV Drawing Tool

# # a. Draw Line (cv2.line())
# background = np.zeros((400, 400, 3)).astype(np.uint8)

# horizontal line (red), y0 = yt
# cv2.line(background, (100, 350), (300, 350), (50, 0, 255), 3, cv2.FILLED)

# vertical line (green), x0 = xt
# cv2.line(background, (50, 100), (50, 300), (25, 255, 0), 20, cv2.LINE_8)

# garis miring (pink)
# cv2.line(background, (250, 300), (230, 100), (255, 0, 255), 5, cv2.LINE_4)

# garis miring (tosca)
# cv2.line(background, (300, 300), (280, 100), (100, 127, 0), 5, cv2.LINE_AA)

# cv2.imshow("Draw Line", background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # b. Draw Rectangle (cv2.rectangle())

# outline color
# cv2.rectangle(background, (15, 25), (200, 150), (0, 0, 255), 5)

# fill color
# cv2.rectangle(background, (210, 50), (270, 270),
#               (0, 200, 255), -1, cv2.LINE_AA)

# cv2.imshow("Draw Rectangle", background)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # c. Draw Circle (cv2.circle())
# img = cv2.imread("data1\\tiga_tepung.jpg")

# circle outline
# cv2.circle(img, (65, 65), 55, (0, 255, 150), 2, cv2.LINE_AA)

# circle fill
# cv2.circle(img, (65, 250), 55, (0, 50, 250), -1, cv2.LINE_AA)

# cv2.imshow("Draw Circle", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# background = np.zeros((200, 600, 3)).astype(np.uint8)
# h, w, c = background.shape

# font_types = [cv2.FONT_HERSHEY_SIMPLEX,
#               cv2.FONT_HERSHEY_PLAIN,
#               cv2.FONT_HERSHEY_DUPLEX,
#               cv2.FONT_HERSHEY_COMPLEX,
#               cv2.FONT_HERSHEY_TRIPLEX,
#               cv2.FONT_HERSHEY_COMPLEX_SMALL,
#               cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
#               cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
#               cv2.FONT_ITALIC]

# texts = ['FONT HERSHEY SIMPLEX',
#          'FONT HERSHEY PLAIN',
#          'FONT HERSHEY DUPLEX',
#          'FONT HERSHEY COMPLEX',
#          'FONT HERSHEY TRIPLEX',
#          'FONT HERSHEY COMPLEX SMALL',
#          'FONT HERSHEY SCRIPT SIMPLEX',
#          'FONT HERSHEY SCRIPT COMPLEX',
#          'FONT ITALIC']

# for text, font_type in zip(texts, font_types):
#     frame = background.copy()
#     cv2.putText(frame, text, (50, 50), font_type,
#                 0.9, (0, 255, 127), 1, cv2.LINE_AA)

#     cv2.imshow("Write Text", frame)
#     cv2.waitKey(2000)  # delay 2 second

# cv2.destroyAllWindows()

# # Mouse Event Click untuk draw line
# x0, y0, xt, yt = 0, 0, 0, 0

# title_window = "Draw Line"
# is_draw = False
# frame = np.ones((400, 400, 3)).astype(np.uint8)*255


# def draw_line(x0, y0, xt, yt):
#     background = frame.copy()
#     cv2.line(background, (x0, y0), (xt, yt), (255, 0, 255), 15, cv2.LINE_AA)
#     cv2.imshow(title_window, background)


# def read_line(event, x, y, flags, param):
#     global x0, y0, xt, yt, is_draw

#     if event == cv2.EVENT_LBUTTONDOWN:
#         x0, y0, xt, yt = x, y, x, y
#         is_draw = True

#     elif event == cv2.EVENT_MOUSEMOVE:
#         xt, yt = x, y

#     elif event == cv2.EVENT_LBUTTONUP:
#         xt, yt = x, y
#         is_draw = False

#     if is_draw:
#         draw_line(x0, y0, xt, yt)


# cv2.namedWindow(title_window)
# cv2.setMouseCallback(title_window, read_line)

# draw_line(x0, y0, xt, yt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Mouse Event Click untuk draw circle
# import math
# x0, y0, r = 0, 0, 0

# title_window = "Draw Circle"
# is_draw = False
# frame = np.ones((400, 400, 3)).astype(np.uint8)*255


# def draw_circle(x0, y0, r):
#     background = frame.copy()
#     cv2.circle(background, (x0, y0), r, (0, 255, 150), 2, cv2.LINE_AA)
#     cv2.imshow(title_window, background)


# def read_circle(event, x, y, flags, param):

#     global x0, y0, r, is_draw

#     if event == cv2.EVENT_LBUTTONDOWN:
#         x0, y0, r = x, y, 0
#         is_draw = True

#     elif event == cv2.EVENT_MOUSEMOVE:
#         r = int(math.sqrt((x - x0)**2 + (y - y0)**2))

#     elif event == cv2.EVENT_LBUTTONUP:
#         r = int(math.sqrt((x - x0)**2 + (y - y0)**2))
#         is_draw = False

#     if is_draw:
#         draw_circle(x0, y0, r)


# cv2.namedWindow(title_window)
# cv2.setMouseCallback(title_window, read_circle)

# draw_circle(x0, y0, r)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Mouse Event Click untuk draw rectangle
# x0, y0, xt, yt = 0, 0, 0, 0

# title_window = "Draw Rectangle"
# is_draw = False
# frame = np.ones((400, 400, 3)).astype(np.uint8)*255


# def draw_rectangle(x0, y0, xt, yt):
#     background = frame.copy()
#     cv2.rectangle(background, (x0, y0), (xt, yt),
#                   (0, 200, 255), -1, cv2.LINE_AA)
#     cv2.imshow(title_window, background)


# def read_rectangle(event, x, y, flags, param):

#     global x0, y0, xt, yt, is_draw

#     if event == cv2.EVENT_LBUTTONDOWN:
#         x0, y0, xt, yt = x, y, x, y
#         is_draw = True

#     elif event == cv2.EVENT_MOUSEMOVE:
#         xt, yt = x, y

#     elif event == cv2.EVENT_LBUTTONUP:
#         xt, yt = x, y
#         is_draw = False

#     if is_draw:
#         draw_rectangle(x0, y0, xt, yt)


# cv2.namedWindow(title_window)
# cv2.setMouseCallback(title_window, read_rectangle)

# draw_rectangle(x0, y0, xt, yt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # # 2. Counter Detection and Countour Drawing

# # A. Method cv2.RETR_EXTERNAL
# img = cv2.imread('pertemuan10\\hierarchy.png')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)

# cv2.imshow("Drawing Contour - Method External", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # B. Method cv2.RETR_TREE
# img = cv2.imread('pertemuan10\\hierarchy.png')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
#     cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)

# cv2.imshow("Drawing Contour - Method TREE", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # C. Method cv2.RETR_CCOMP
# img = cv2.imread('pertemuan10\\hierarchy.png')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
#     cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)

# cv2.imshow("Drawing Contour - Method CCOMP", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # C. Method cv2.RETR_LIST
# img = cv2.imread('pertemuan10\\hierarchy.png')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
#     cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)

# cv2.imshow("Drawing Contour - Method CCOMP", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # D. Contour Feature

# img = cv2.imread('pertemuan10\\noisy_text.png')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
# cv2.drawContours(img, cnt, -1, (255, 0, 255), 3)
# perimeter = cv2.arcLength(cnt,True)
# print("keliling : %d pixel" % perimeter)

# cv2.imshow("Drawing Contour", img)
# cv2.imshow("Binary", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Contour Area (luasan)
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     print("luas : %d pixel" % area)

# # Contour Perimeter (keliling)
# for cnt in contours:
#     perimeter = cv2.arcLength(cnt, True)
#     print("keliling : %d pixel" % perimeter)

# # Contour Approximation
# for cnt in contours:
#     epsilon = 0.001*cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     cv2.polylines(img, [approx], True, (0, 255, 255), 3)

# cv2.imshow("Contour Approximation", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Trackbar change epsilon to arc length portion on Contpur Approximation
# max_value = 10000
# default_value = 10

# title_window = "Contour Approximation"


# def on_trackbar(val):
#     if val > 0:
#         frame = img.copy()
#         for cnt in contours:
#             epsilon = (1/val)*cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, epsilon, True)
#             cv2.polylines(frame, [approx], True, (0, 255, 255), 3)
#             cv2.imshow(title_window, frame)


# img = cv2.imread('pertemuan10\\noisy_text.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('divisor', title_window,
#                    default_value, max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Draw Straight Bounding Rectangle for each contour using cv2.rectangle()

# img = cv2.imread('pertemuan10\\noisy_text.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# frame = img.copy()
# for cnt in contours:
#     rect = cv2.boundingRect(cnt)
#     x, y, w, h = rect
#     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

# cv2.imshow("Contour Approximation &", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Crop each contour from input image and display

# img = cv2.imread('pertemuan10\\noisy_text.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# frame = img.copy()
# for i, cnt in enumerate(contours):
#     rect = cv2.boundingRect(cnt)
#     x, y, w, h = rect
#     roi = frame[y:y+h, x:x+w]
#     cv2.imshow("Contour-%d" % i, roi)

# roi = []
# for i, cnt in enumerate(contours):
#     rect = cv2.boundingRect(cnt)
#     x, y, w, h = rect
#     roi.append(frame[y:y+h, x:x+w])
#     cv2.imshow("Contour-%d" % i, cnt)

# for i, item in enumerate(roi):
#     cv2.imshow("image roi - %d" % i, item)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Crop foto KTP
# img = cv2.imread("pertemuan10\\ktp2.jpg")
# h_img, w_img, c = img.shape

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh, binary = cv2.threshold(
#     gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# contours, hierarchy = cv2.findContours(
#     binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
#     rect = cv2.boundingRect(cnt)
#     x, y, w, h = rect

#     aspectRatio = w/h
#     heightRatio = h / h_img

#     keepAspectRatio = aspectRatio > 0.7 and aspectRatio < 0.9
#     keepHeighRatio = heightRatio > 0.4 and heightRatio < 0.5

#     if keepAspectRatio and keepHeighRatio:
#         roi = img[y:y+h, x:x+w]
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

#         cv2.imshow("Binary", binary)
#         cv2.imshow("Cropped Photo", roi)

# cv2.imshow("Detected Photo", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Crop character plat nomor dari gambar
# img = cv2.imread("pertemuan10\\plat-nomor-5.jpg")

# ratio = 350.0/img.shape[0]
# img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
# h_img, w_img, c = img.shape

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# thresh, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(
#     binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# for i, cnt in enumerate(contours):
#     area = cv2.contourArea(cnt)
#     rect = cv2.boundingRect(cnt)
#     x, y, w, h = rect

#     aspectRatio = w/h
#     heightRatio = h / h_img
#     Next, Previous, First_Child, Parent = hierarchy[0][i]

#     keepAspectRatio = aspectRatio > 0.2 and aspectRatio < 0.7
#     keepHeightRatio = heightRatio > 0.25 and heightRatio < 0.5

#     # remove small contour
#     if area < 5.0 or w < 3.0 or h < 3.0:
#         continue

#     if keepAspectRatio and keepHeightRatio and Parent == -1 and Next != -1 and Previous != -1:
#         roi = img[y:y+h, x:x+w].copy()
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

#         cv2.imshow("Cropped Photo - %d" % i, roi)
#         cv2.imshow("Binary", binary)
#         cv2.imshow("Detected Photo", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

ratio = 350.0/img.shape[0]
img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
h_img, w_img, c = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh, binary_tzr = cv2.threshold(
    gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
thresh, binary = cv2.threshold(binary_tzr, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    rect = cv2.boundingRect(cnt)
    x, y, w, h = rect

    aspectRatio = w/h
    heightRatio = h / h_img
    Next, Previous, First_Child, Parent = hierarchy[0][i]

    keepAspectRatio = aspectRatio > 0.2 and aspectRatio < 0.7
    keepHeighRatio = heightRatio > 0.25 and heightRatio < 0.5

    # remove small contour
    if area < 5.0 or w < 3.0 or h < 3.0:
        continue

    if keepAspectRatio and keepHeighRatio and Parent == -1 and Next != -1 and Previous != -1:
        roi = img[y:y+h, x:x+w].copy()
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

        cv2.imshow("Cropped Photo - %d" % i, roi)
        cv2.imshow("Binary", binary)
        cv2.imshow("Detected Photo", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
