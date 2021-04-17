import cv2
import numpy as np

# # 1. Image Smoothing

# a. cv2.blur()

# img = cv2.imread('pertemuan9\\noisy_mri.jpg')

# blur = cv2.blur(img, (5, 5))

# cv2.imshow("Original Image", img)
# cv2.imshow("Blur Image", blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Atur Kernel Size Menggunakan Trackbar
# max_value = 10
# default_value = 5

# title_window = "Blur Image"


# def on_trackbar(val):
#     if val > 0:
#         blur = cv2.blur(img, (val, val), (-1, -1))
#         cv2.imshow(title_window, blur)


# img = cv2.imread('pertemuan9\\noisy_mri.jpg')

# cv2.namedWindow(title_window)
# cv2.createTrackbar('kernel', title_window,
#                    default_value, max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # menampilkan cv2.blur dan cv2.GaussianBlur() dalam 1 window
# max_value = 10
# default_value = 5

# title_window = "Gaussian Blur Image"


# def on_trackbar(val):
#     # filter value > 0 and odd number
#     if val > 0 and val % 2 == 1:
#         blur1 = cv2.blur(img, (val, val), (-1, -1))
#         blur2 = cv2.GaussianBlur(img, (val, val), 0, 0)

#         frame = np.zeros((h, w*2, c)).astype(np.uint8)

#         frame[0:h, 0:w] = blur1
#         frame[0:h, w:2*w] = blur2
#         cv2.imshow(title_window, frame)


# img = cv2.imread('pertemuan9\\noisy_mri.jpg')
# h, w, c = img.shape

# cv2.namedWindow(title_window)
# cv2.createTrackbar('kernel', title_window,
#                    default_value, max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Coba Gaussian Blur (2)
# max_value = 10
# default_value = 5

# title_window = "Gaussian Blur Image"


# def on_trackbar(val):
#     # filter value > 0 and odd number
#     if val > 0 and val % 2 == 1:
#         blur1 = cv2.blur(img, (val, val), (-1, -1))
#         blur2 = cv2.GaussianBlur(img, (val, val), 0, 0)

#         frame = np.zeros((h*2, w, c)).astype(np.uint8)

#         frame[0:h, 0:w] = blur1
#         frame[h:2*h, 0:w] = blur2
#         cv2.imshow(title_window, frame)

# img = cv2.imread('pertemuan9\\noisy_mri.jpg')
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# h, w, c = img.shape

# cv2.namedWindow(title_window)
# cv2.createTrackbar('kernel', title_window, default_value, max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 3. Image Binarization

# # a. Simple Thresholding
# img = cv2.imread('pertemuan9\\noisy_mri.jpg')

# convert to grayscale
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding
# cv2.thresholding(<image>, threshold_value, max_value, threshold_method)
# ret1, thresh1 = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
# ret2, thresh2 = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
# ret3, thresh3 = cv2.threshold(img, 230, 255, cv2.THRESH_TRUNC)
# ret4, thresh4 = cv2.threshold(img, 230, 255, cv2.THRESH_TOZERO)
# ret5, thresh5 = cv2.threshold(img, 230, 255, cv2.THRESH_TOZERO_INV)

# show image
# cv2.imshow("grayscale image", img)
# cv2.imshow("Threshold Binary", thresh1)
# cv2.imshow("Threshold Binary Inv", thresh2)
# cv2.imshow("Threshold Trunc", thresh3)
# cv2.imshow("Threshold To Zero", thresh4)
# cv2.imshow("Threshold To Zero Inv", thresh5)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # menambahkan trackbar untuk contrl val threshold
# max_value = 255
# default_value = 127

# title_window = "Trackbar"


# def on_trackbar(val):
#     ret1, thresh1 = cv2.threshold(img, val, max_value, cv2.THRESH_BINARY)
#     ret2, thresh2 = cv2.threshold(img, val, max_value, cv2.THRESH_BINARY_INV)
#     ret3, thresh3 = cv2.threshold(img, val, max_value, cv2.THRESH_TRUNC)
#     ret4, thresh4 = cv2.threshold(img, val, max_value, cv2.THRESH_TOZERO)
#     ret5, thresh5 = cv2.threshold(img, val, max_value, cv2.THRESH_TOZERO_INV)

#     cv2.imshow("Original Image", img)
#     cv2.imshow("Thresholded Binary", thresh1)
#     cv2.imshow("Thresholded Binary Inv", thresh2)
#     cv2.imshow("Thresholded Trunc", thresh3)
#     cv2.imshow("Thresholded To Zero", thresh4)
#     cv2.imshow("Thresholded To Zero Inv", thresh5)


# img = cv2.imread('pertemuan9\\number_plate.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('threshold', title_window,
#                    default_value, max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # combine dalam 1 frame untuk thresholding
# max_value = 255
# default_value = 127

# title_window = "Simple Thresholding"


# def on_trackbar(val):
#     ret1, thresh1 = cv2.threshold(img, val, max_value, cv2.THRESH_BINARY)

#     frame = np.zeros((h, w*2)).astype(np.uint8)

#     frame[0:h, 0:w] = img
#     frame[0:h, w:2*w] = thresh1

#     cv2.imshow(title_window, frame)


# img = cv2.imread('pertemuan9\\number_plate.jpg')
# h, w, c = img.shape
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('threshold', title_window,
#                    default_value, max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # tugas buatlah macam-macam thresholding menjadi 1 window
# max_value = 255
# default_value = 127

# title_window = "Simple Thresholding"


# def on_trackbar(val):
#     ret1, thresh1 = cv2.threshold(img, val, max_value, cv2.THRESH_BINARY)
#     ret2, thresh2 = cv2.threshold(img, val, max_value, cv2.THRESH_BINARY_INV)
#     ret3, thresh3 = cv2.threshold(img, val, max_value, cv2.THRESH_TRUNC)
#     ret4, thresh4 = cv2.threshold(img, val, max_value, cv2.THRESH_TOZERO)
#     ret5, thresh5 = cv2.threshold(img, val, max_value, cv2.THRESH_TOZERO_INV)

#     frame = np.zeros((h*2, w*3)).astype(np.uint8)

#     frame[0:h, 0:w] = img
#     frame[0:h, w:w*2] = thresh1
#     frame[0:h, w*2:w*3] = thresh2
#     frame[h:h*2, 0:w] = thresh3
#     frame[h:h*2, w:w*2] = thresh4
#     frame[h:h*2, w*2:w*3] = thresh5

#     cv2.imshow(title_window, frame)


# # Thresh Otsu

max_value = 255
default_value = 127

title_window = "Trackbar"


def on_trackbar(val):
    # global thresholding
    ret1, th1 = cv2.threshold(gray, val, max_value, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(gray, val, max_value,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray, (3, 3), 0, 0)
    ret3, th3 = cv2.threshold(blur, val, max_value,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # show image
    cv2.imshow("Original Image", img)
    cv2.imshow("Global Thresholding", th1)
    cv2.imshow("Otsu's Thresholding", th2)
    cv2.imshow("Otsu's Thresholding after Gaussian Filter", th3)


img = cv2.imread('pertemuan9\\number_plate.jpg')
# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow(title_window)
cv2.createTrackbar('threshold', title_window,
                   default_value, max_value, on_trackbar)

on_trackbar(default_value)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 3. Edge Detection (Canny Edge Detection)
# img = cv2.imread('pertemuan9\\blocks.jpg')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.canny(<image>, threshold_min, threshold_max)
# edged = cv2.Canny(gray, 100, 250)

# show image
# cv2.imshow("Original Image", img)
# cv2.imshow("Edged Image", edged)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Menggunkan trackbar untuk mengubah nilai min, max pada cv2.canny()
# max_value = 255
# default_value = 127

# current_min = 0
# current_max = 255

# title_window = "Edge Detection"


# def on_trackbar_min(val):
#     global current_min
#     current_min = val
#     edged = cv2.Canny(gray, current_min, current_max)
#     cv2.imshow(title_window, edged)


# def on_trackbar_max(val):
#     global current_max
#     current_max = val
#     edged = cv2.Canny(gray, current_min, current_max)
#     cv2.imshow(title_window, edged)


# img = cv2.imread('pertemuan9\\blocks.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('min', title_window, default_value,
#                    max_value, on_trackbar_min)
# cv2.createTrackbar('max', title_window, default_value,
#                    max_value, on_trackbar_max)

# on_trackbar_min(0)
# cv2.imshow("Original Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # # Tugas
# Apply Binary thresholding sebelum diterapkan pada edge detector
# Tambahkan trackbar untuk mengatur threshold value Binary Thresholding
# set min dan max edge detector ke 127 dan 200 (cv2.Canny(binary_img, 127, 200))
# max_value = 255
# default_value = 127

# title_window = "Edge Detection"

# img = cv2.imread('pertemuan9\\blocks.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# def on_trackbar_thresh(val):
#     ret, binary_img = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
#     edged = cv2.Canny(binary_img, 127, 200)

#     h, w = binary_img.shape
#     frame = np.zeros((h*2, w)).astype(np.uint8)

#     frame[:h, :w] = binary_img
#     frame[h:h*2, :w] = edged

#     cv2.imshow(title_window, frame)


# cv2.namedWindow(title_window)
# cv2.createTrackbar('thresh', title_window, default_value,
#                    max_value, on_trackbar_thresh)

# on_trackbar_thresh(127)
# cv2.imshow("Original Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
