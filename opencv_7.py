import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Gaussian Pyramid
# # Downscale & Upscaling lena.jpg
# img = cv2.imread("pertemuan13\\lena.jpg")
# img_PD = cv2.pyrDown(img)

# cv2.imshow("pyramid downscale 1/4", img_PD)
# cv2.imshow("original", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img_PU = cv2.pyrUp(img_PD)

# cv2.imshow("pyramid upscale 4x", img_PU)
# cv2.imshow("original", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Multiple downscale lena.jpg
# img = cv2.imread("pertemuan13\\lena.jpg")
# h, w, c = img.shape

# # print("image - %d : %d,%d" % (0, h, w))
# # cv2.imshow("image - %d" % 0, img)

# for i in range(1, 10):
#     img = cv2.pyrDown(img)
#     h, w, c = img.shape
#     if h < 100 and w < 100:
#         break
#     # print("image - %d : %d,%d" % (i, h, w))
#     # cv2.imshow("image - %d" % i, img)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Multiple Upscale lena.jpg
# img = cv2.imread("pertemuan13\\lena.jpg")
# h, w, c = img.shape

# print("image - %d : %d,%d" % (0, h, w))
# cv2.imshow("image - %d" % 0, img)

# for i in range(1, 10):
#     img = cv2.pyrUp(img)
#     h, w, c = img.shape
#     if h > 512 and w > 512:
#         break
#     print("image - %d : %d,%d" % (i, h, w))
#     cv2.imshow("image - %d" % i, img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # B. Basic Image Operation Review

# # 1. Image Addition // method : cv2.add(img1, img2)
# img1 = np.ones((5, 5, 3)).astype(np.uint8)*50
# img2 = np.ones((5, 5, 3)).astype(np.uint8)*127

# out = cv2.add(img1, img2)

# print(out[:, :, 0])
# plt.imshow(out[:, :, ::-1])
# plt.show()

# # C. Image Substraction
# # method : cv2.subtract(img1, img2)
# img1 = np.ones((5, 5, 3)).astype(np.uint8)*150
# img2 = np.ones((5, 5, 3)).astype(np.uint8)*200

# out = cv2.subtract(img1, img2)

# print(out[:, :, 0])
# plt.imshow(out[:, :, ::-1])

# img1 = cv2.imread("pertemuan13\\apple.jpg")
# img2 = cv2.imread("pertemuan13\\orange.jpg")

# out = cv2.subtract(img1, img2)
# blend = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# # plt.imshow(out[:, :, ::-1])
# plt.imshow(blend[:, :, ::-1])
# plt.show()

# img1 = cv2.imread("pertemuan13\\apple.jpg")
# img2 = cv2.imread("pertemuan13\\orange.jpg")
# h, w, c = img1.shape

# frame = np.zeros((h, w*2, c)).astype(np.uint8)
# frame[:, :w] = img1
# frame[:, w:w*2] = img2

# plt.imshow(frame[:, :, ::-1])
# plt.show()

# # D. Merging image (numpy)

# # horizontal merging // np.hstack((img1, img2))
# img1 = cv2.imread("pertemuan13\\apple.jpg")
# img2 = cv2.imread("pertemuan13\\orange.jpg")

# out = np.hstack((img1, img2))

# plt.imshow(out[:, :, ::-1])
# plt.show()

# out = np.vstack((img1, img2))

# plt.imshow(out[:,:, ::-1])
# plt.show()

# # E. Laplacian Pyramid

# GP_0 = cv2.imread("pertemuan13\\lena.jpg")  # 512x512
# GP_1 = cv2.pyrDown(GP_0)  # 256x256
# LP_0 = cv2.subtract(GP_0, cv2.pyrUp(GP_1))  # 512x512

# cv2.imshow("GP 0", GP_0)
# cv2.imshow("GP 1", GP_1)
# cv2.imshow("LP 0", LP_0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 1. Multiple Laplacian Layer

# img = cv2.imread("pertemuan13\\lena.jpg")

# GP = img.copy()
# print(GP.shape)
# GP_list = [GP]  # Gaussina Pyramid `pyrDown` list image
# for i in range(0, 3):
#     GP = cv2.pyrDown(GP)
#     GP_list.append(GP)
#     print(GP.shape)

# for i, GP in enumerate(GP_list):
#     cv2.imshow("GP image - %d" % i, GP)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# LP_list = [GP_list[-1]]  # laplacian Pyramid list image
# print(LP_list[0].shape)
# for i in range(3, 0, -1):
#     LP = cv2.subtract(GP_list[i-1], cv2.pyrUp(GP_list[i]))
#     LP_list.append(LP)
#     print(LP.shape)

# for i, LP in enumerate(LP_list):
#     cv2.imshow("LP image - %d" % i, LP)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Penerapan Image Pyramid untuk Image Stitching

# Simple Image Stitching||tetapi mungkin tidak terlihat bagus karena diskontinuitas di antara gambar
# img1 = cv2.imread("pertemuan13\\apple.jpg")
# img2 = cv2.imread("pertemuan13\\orange.jpg")

# h, w, c = img1.shape
# print(h, w, c)

# output = np.zeros((h, w, c)).astype(np.uint8)
# output[:, :w//2] = img1[:, :w//2]
# output[:, w//2:] = img2[:, w//2:]

# cv2.imshow("stiching image", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Stitching using Image Pyramid

# HSTACK
# img1 = cv2.imread("pertemuan13\\apple.jpg")
# img2 = cv2.imread("pertemuan13\\orange.jpg")

# GP1 = img1.copy()
# GP1_list = [GP1]
# for i in range(6):
#     GP1 = cv2.pyrDown(GP1)
#     GP1_list.append(GP1)

# GP2 = img2.copy()
# GP2_list = [GP2]
# for i in range(6):
#     GP2 = cv2.pyrDown(GP2)
#     GP2_list.append(GP2)

# LP1_list = [GP1_list[-1]]
# for i in range(6, 0, -1):
#     LP1 = cv2.subtract(GP1_list[i-1], cv2.pyrUp(GP1_list[i]))
#     LP1_list.append(LP1)

# LP2_list = [GP2_list[-1]]
# for i in range(6, 0, -1):
#     LP2 = cv2.subtract(GP2_list[i-1], cv2.pyrUp(GP2_list[i]))
#     LP2_list.append(LP2)

# LS = []
# for L1, L2 in zip(LP1_list, LP2_list):
#     h, w, c = L1.shape
#     L = np.hstack((L1[:, :w//2], L2[:, w//2:]))
#     LS.append(L)

# output = LS[0]
# for i in range(1, 7):
#     output = cv2.add(cv2.pyrUp(output), LS[i])


# cv2.imshow("stiching image pyramid", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# VSTACK
# img1 = cv2.imread("pertemuan13\\apple.jpg")
# img2 = cv2.imread("pertemuan13\\orange.jpg")

# GP1 = img1.copy()
# GP1_list = [GP1]
# for i in range(6):
#     GP1 = cv2.pyrDown(GP1)
#     GP1_list.append(GP1)

# GP2 = img2.copy()
# GP2_list = [GP2]
# for i in range(6):
#     GP2 = cv2.pyrDown(GP2)
#     GP2_list.append(GP2)

# LP1_list = [GP1_list[-1]]
# for i in range(6, 0, -1):
#     LP1 = cv2.subtract(GP1_list[i-1], cv2.pyrUp(GP1_list[i]))
#     LP1_list.append(LP1)

# LP2_list = [GP2_list[-1]]
# for i in range(6, 0, -1):
#     LP2 = cv2.subtract(GP2_list[i-1], cv2.pyrUp(GP2_list[i]))
#     LP2_list.append(LP2)

# LS = []
# for L1, L2 in zip(LP1_list, LP2_list):
#     h, w, c = L1.shape
#     L = np.vstack((L1[:w//2, :], L2[w//2:, :]))
#     LS.append(L)

# output = LS[0]
# for i in range(1, 7):
#     output = cv2.add(cv2.pyrUp(output), LS[i])


# cv2.imshow("stiching image pyramid", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # # Geometric Transform

# # Rotation 1
# img = cv2.imread("pertemuan13\\lena.jpg")
# h, w, c = img.shape

# center = (w //2, h // 2)
# M = cv2.getRotationMatrix2D(center, 45, 1.0)
# rotated = cv2.warpAffine(img, M, (w, h))

# cv2.imshow("rotated image", rotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # trackbar to rotate Image
# max_value = 360
# default_value = 10

# title_window = "Rotate Image"


# def on_trackbar(val):
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, val, 1.0)
#     rotated = cv2.warpAffine(img, M, (w, h))
#     cv2.imshow(title_window, rotated)


# img = cv2.imread('pertemuan13\\lena.jpg')
# w, h, c = img.shape

# cv2.namedWindow(title_window)
# cv2.createTrackbar('angel', title_window, default_value,
#                    max_value, on_trackbar)

# on_trackbar(default_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Image Translation

# img = cv2.imread('pertemuan13\\lena.jpg')
# h, w, c = img.shape
# M = np.float32([[1, 0, 0],
#                 [0, 1, -100]])

# translated = cv2.warpAffine(img, M, (w, h))

# cv2.imshow("Image Translation", translated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Affine Transform

# img = cv2.imread('pertemuan13\\lena.jpg')
# h, w, c = img.shape

# pts1 = np.float32([[0, 0], [w, 0], [0, h]])  # source
# pts2 = np.float32([[0, 100], [400, 50], [50, 250]])  # destination

# M = cv2.getAffineTransform(pts1, pts2)
# warp_dst = cv2.warpAffine(img, M, (w, h))

# cv2.imshow("Warp Transform", warp_dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# pts1 = np.float32([[0, 100], [400, 50], [50, 250]])  # source
# pts2 = np.float32([[0, 0], [w, 0], [0, h]])  # dst

# M = cv2.getAffineTransform(pts1, pts2)

# warp_dst2 = cv2.warpAffine(warp_dst, M, (w, h))

# cv2.imshow("Original", img)
# cv2.imshow("Warp Transform", warp_dst)
# cv2.imshow("Warp Transform 2", warp_dst2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Affine Transformation Translation on click
# pts1 = []
# pts2 = []
# windowName = "Affine Transform by Mouse Click"
# is_edit = False

# def affine_transform(event, x, y, flags, param):

#     global pts1, pts2, is_edit, frame

#     if event == cv2.EVENT_RBUTTONDOWN:
#         is_edit = True
#         pts1 = []
#         pts2 = []
#         frame = img.copy()

#     if event == cv2.EVENT_LBUTTONDOWN and is_edit:
#         if len(pts1) < 3:
#             pts1.append([x, y])
#             cv2.circle(frame, (x,y), 6, (255,255,0), -1)

#         elif len(pts2) < 3:
#             if len(pts2) == 0:
#                 frame = np.zeros((frame.shape)).astype(np.uint8)

#             pts2.append([x, y])
#             cv2.circle(frame, (x,y), 6, (0,255,255), -1)

#         else:
#             w, h, c = frame.shape
#             pts1 = np.float32(pts1)
#             pts2 = np.float32(pts2)
#             M = cv2.getAffineTransform(pts1, pts2)
#             frame = cv2.warpAffine(img.copy(), M, (w,h))
#             is_edit = False

# cv2.namedWindow(windowName)
# cv2.setMouseCallback(windowName, affine_transform)

# img = cv2.imread("pertemuan13\\lena.jpg")
# frame = img.copy()
# while True:
#     cv2.imshow(windowName, frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cv2.destroyAllWindows()

# # # Perspective Transformation

# # A. Gambar Sudoku
# img = cv2.imread('pertemuan13\\sudoku.jpg')
# h, w, c = img.shape

# # tl, tr, br, bl
# pts1 = np.float32([[56, 65], [368, 52], [389, 390], [28, 387]])
# pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

# M = cv2.getPerspectiveTransform(pts1, pts2)

# output = cv2.warpPerspective(img, M, (w, h))

# for x, y in pts1.astype(np.uint16):
#     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)

# cv2.imshow("Original Image", img)
# cv2.imshow("Perspective Transform Image", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # using mouse click to put the coordinate
# def order_points(pts):
#     # rect = np.zeros((4, 2), dtype="float32")
#     # s = pts.sum(axis=1)
#     # rect[0] = pts[np.argmin(s)]
#     # rect[2] = pts[np.argmax(s)]
#     # diff = np.diff(pts, axis=1)
#     # rect[1] = pts[np.argmin(diff)]
#     # rect[3] = pts[np.argmax(diff)]
#     # #tl, tr, br, bl
#     # return rect


# tl, tr, br, bl
# pts1 = []
# pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
# windowName = "Perspective Transform by Mouse Click"
# is_edit = False


# def perspective_transform(event, x, y, flags, param):

#     global pts1, is_edit, frame

#     if event == cv2.EVENT_RBUTTONDOWN:
#         is_edit = True
#         pts1 = []
#         frame = img.copy()

#     if event == cv2.EVENT_LBUTTONDOWN and is_edit:
#         if len(pts1) < 4:
#             pts1.append([x, y])
#             cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

#         else:
#             # pts1 = order_points(np.array(pts1))
#             #pts1 = np.float32(pts1)
#             M = cv2.getPerspectiveTransform(pts1, pts2)
#             output = cv2.warpPerspective(img, M, (300, 300))
#             cv2.imshow("Perpective Transform Result", output)
#             is_edit = False


# cv2.namedWindow(windowName)
# cv2.setMouseCallback(windowName, perspective_transform)

# img = cv2.imread("pertemuan13\\sudoku.jpg")
# frame = img.copy()
# while True:
#     cv2.imshow(windowName, frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cv2.destroyAllWindows()

# # Penerapan Affine Transform untuk crop foto ktp dengan perspective transform

def get_contours(img):
    # # First make the image 1-bit and get contours
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours that are too large or small
    h, w, c = img.shape
    size = h*w
    contours = [cc for cc in contours if contourOK(cc, size)]
    return contours


def contourOK(cc, size=1000000):
    x, y, w, h = cv2.boundingRect(cc)
    if w < 50 or h < 50:
        return False  # too narrow or wide is bad
    area = cv2.contourArea(cc)
    return area < (size*0.25) and area > 4000


def find_4_coord(contours):
    perimeter = []
    for cc in contours:
        perimeter = cv2.approxPolyDP(cc, 0.09 * cv2.arcLength(cc, True), True)
        return np.array(perimeter[:, 0, :])


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    print(tl, tr, br, bl)

    for pt in rect.astyp(np.uint16):
        cv2.circle(img, (pt[0], pt[1]), 4, (0, 255, 255), -1)
