import cv2
import numpy as np
import matplotlib.pyplot as plt

# # 1. Basic Morphological Transform (Eroding & Dilating)

# frame = np.zeros((20, 20), np.uint8)
# cv2.circle(frame, (5, 5), 3, (255, 255, 255), -1, cv2.LINE_AA)
# cv2.circle(frame, (14, 14), 3, (255, 255, 255), -1, cv2.LINE_AA)
# print(plt.imshow(frame, cmap="gray"))
# plt.show()

# kernel = np.ones((3, 3), np.uint8)
# print(kernel)

# dilate = cv2.dilate(frame.copy(), kernel, iterations=1)
# plt.imshow(dilate, cmap="gray")
# plt.show()

# # A. Denoising Citra MRI menggunakan Eroding
# Penggunaan cv2.erode() + square kernel mirip dengan cv2.blur()
# img = cv2.imread("pertemuan12\\noisy_mri.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = np.ones((3, 3), np.uint8)
# erosion = cv2.erode(gray, kernel, iterations=1)

# cv2.imshow("Erosion", erosion)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # B. Fixing Broken Character menggunakan Dilating
# img = cv2.imread('pertemuan12\\Broker_Char.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# kernel = np.ones((5, 5), np.uint8)
# dilating = cv2.dilate(thresh, kernel, iterations=5)

# cv2.imshow("Dilating", dilating)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # C. Menambahkan Trackbar Fixing Broken Character menggunakan Dilating
# iteration = 1
# kernel_size = 3
# title_window = "Dilation Image"


# def on_trackbar_iteration(val):
#     if val > 0:
#         global iteration
#         iteration = val
#         kernel = np.ones((kernel_size, kernel_size), np.uint8)
#         dilating = cv2.dilate(thresh, kernel, iterations=iteration)
#         cv2.imshow(title_window, dilating)


# def on_trackbar_kernel_size(val):
#     if val > 0:
#         global kernel_size
#         kernel_size = val
#         kernel = np.ones((kernel_size, kernel_size), np.uint8)
#         dilating = cv2.dilate(thresh, kernel, iterations=iteration)


# img = cv2.imread('pertemuan12\\Broker_Char.png')
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# cv2.namedWindow(title_window)
# cv2.createTrackbar('kernel', title_window, 3, 10, on_trackbar_kernel_size)
# cv2.createTrackbar('iteration', title_window, 1, 10, on_trackbar_iteration)

# on_trackbar_kernel_size(3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # TASK
# img = cv2.imread('pertemuan12\\Noised_Broken_Char.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# kernel = np.ones((3, 3), np.uint8)
# erosion = cv2.erode(thresh, kernel, iterations=3)
# dilating = cv2.dilate(erosion, kernel, iterations=9)
# normalize = cv2.erode(dilating, kernel, iterations=7)

# cv2.imshow("Erosion", erosion)
# cv2.imshow("Dilating", dilating)
# cv2.imshow("Original", img)
# cv2.imshow("normalize", normalize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 2. Advance Morphological Transform (Opening, Closing, & Morphological Gradient)

# # A. Implementation Opening on Noisy MRI Image
# img = cv2.imread('pertemuan12\\noisy_mri.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

# cv2.imshow("Opening", opening)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # B. Implementation Closing on Broken Char Image
# img = cv2.imread('pertemuan12\\Broken_Char_2.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# kernel = np.ones((3, 3), np.uint8)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)

# cv2.imshow("Closing", closing)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # C. Implementation Opening and Closing on Noised & Broken Char Image
# img = cv2.imread('pertemuan12\\Noised_Broken_Char.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=7)

# cv2.imshow("Opening", opening)
# cv2.imshow("Closing", closing)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # D. Morphological Gradient
# img = cv2.imread('pertemuan12\\Char.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# kernel = np.ones((3, 3), np.uint8)

# # coba gradient pakai cara dilasi dan erosi
# erosion = cv2.erode(thresh, kernel, iterations=1)
# dilation = cv2.dilate(thresh, kernel, iterations=1)
# gradient1 = dilation - erosion

# # morphological gradient
# gradient2 = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel, iterations=1)

# cv2.imshow("Gradient", gradient1)
# cv2.imshow("Morphological Gradient", gradient2)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # E. Implementasi Morphological Gradient | Tackling non-uniform illumination in images

# Using Simple Thresholding
# img = cv2.imread('pertemuan12\\StrukBelanja.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

# cv2.imshow("Otsu", thresh)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Menggunakan Morphological Gradient
# img = cv2.imread('pertemuan12\\StrukBelanja.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = np.ones((2, 2), np.uint8)

# gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=1)

# cv2.imshow("Morphological Gradient", gradient)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Implementasi Pada Fingerprint image with non-uniform illumination
# img = cv2.imread('pertemuan12\\Fingerprint.png')
# img = cv2.bitwise_not(img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

# kernel = np.ones((3, 3), np.uint8)
# gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=1)

# cv2.imshow("Morphological Gradient", gradient)
# cv2.imshow("To Zero", thresh)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Deteksi Contour Plat Nomor
# img = cv2.imread('pertemuan12\\number_plate.jpg')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = np.ones((3, 3), np.uint8)
# gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=1)
# ret, thresh = cv2.threshold(gradient, 100, 255, cv2.THRESH_TOZERO)
# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
#     cv2.drawContours(img, [cnt], -1, (0, 0, 255), 1)

# cv2.imshow("Morphological Gradient", gradient)
# cv2.imshow("Edge - Thresholding", thresh)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Morhological Operation | Structuring Element
# print(cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
# print(cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)))
# print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

# img = cv2.imread("pertemuan12\\Broker_Char.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

# cv2.imshow("Closing", closing)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(kernel)

# # Removing staff line from musical sheet (no balok)
# img = cv2.imread("pertemuan12\\not_balok.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# bitwise = cv2.bitwise_not(gray)

# # cv2.imshow("Gray", gray)
# # cv2.imshow("hasil", bitwise)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# ret, thresh = cv2.threshold(bitwise, 100, 255, cv2.THRESH_BINARY)

# cv2.imshow("Binary", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Detect Line on Image
# h, w, c = img.shape
# horizontal_size = w//30

# horizontalStructure = cv2.getStructuringElement(
#     cv2.MORPH_RECT, (horizontal_size, 1))

# print(horizontalStructure)
# print(horizontalStructure.shape)

# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
#                            horizontalStructure, iterations=1)

# cv2.imshow("Detected Staff Line | Opening", opening)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Segmentasi Not Balok

# vertical_size = h // 30

# verticalStructure = cv2.getStructuringElement(
#     cv2.MORPH_RECT, (1, vertical_size))

# # print(verticalStrcuture)

# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
#                            verticalStructure, iterations=1)

# # cv2.imshow("Detected Item | Opening", opening)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# contours, hierarchy = cv2.findContours(
#     opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
#     cv2.drawContours(img, [cnt], -1, (0, 0, 255), 1)

# cv2.imshow("Musical Sheet Segmentation", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # extract line & charcater dari gambar handwritting.jpg
img = cv2.imread("pertemuan12\\handwritting.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bitwise = cv2.bitwise_not(gray)

ret, thresh = cv2.threshold(
    bitwise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

detected_line = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, horizontalStructure, iterations=1)

# print(horizontalStructure)
# print(horizontalStructure.shape)

# cv2.imshow("Detected Line", detected_line)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))

detected_char = cv2.morphologyEx(
    bitwise, cv2.MORPH_OPEN, verticalStructure, iterations=1)

# print(verticalStructure)
# print(verticalStructure.shape)

cv2.imshow("Detected Char", detected_char)
cv2.imshow("Detected Line", detected_line)
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
