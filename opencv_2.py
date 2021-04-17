import numpy as np
import cv2

img = cv2.imread('data1\\tiga_tepung.jpg')
# print(img.shape)

# 1. Image Crop

# crop image[y_min:y_max , x_min:x_max]
# img_crop = img[75:176, 150:370]

# show image
# cv2.imshow('cropped image', img_crop)
# cv2.imshow('original image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# setup titik crop melalui input()
# h, w, c = img.shape

# y_min = int(input("masukkan nilai y_min [0 - %d] :" % h))
# y_max = int(input("masukkan nilai y_max [%d - %d ] :" % (y_min, h)))
# x_min = int(input("masukkan nilai x_min [0 - %d] :" % w))
# x_max = int(input("masukkan nilai x_max [%d - %d ] :" % (x_min, w)))

# img_crop = img[y_min:y_max, x_min:x_max]

# show image
# cv2.imshow('cropped image', img_crop)
# cv2.imshow('original image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Menggunakan Mouse event click pada OpenCV
# setup titik crop melalui mouse click
# review global variable
# x = 200


# def hitung(flag):
#     if flag == "set":
#         global x
#         x = 400

#     if flag == "print":
#         print(x)


# print(x)
# hitung("set")
# hitung("print")
# print(x)

# menggunakan mouse event pada click
# x_start, x_end, y_start, y_end = 0, 0, 0, 0
# windowName = "Original Image"

# image = cv2.imread('data1\\tiga_tepung.jpg')


# def crop_image(event, x, y, flags, param):

#     global x_start, x_end, y_start, y_end

#     # if the left mouse button clicked
#     if event == cv2.EVENT_LBUTTONDOWN:
#         x_start, x_end, y_start, y_end = x, x, y, y

#     # mouse is moving
#     elif event == cv2.EVENT_MOUSEMOVE:
#         x_end, y_end = x, y

#     # if the left mouse button was released
#     elif event == cv2.EVENT_LBUTTONUP:
#         cropped_img = image[y_start:y_end, x_start:x_end]
#         cv2.imshow("Cropped Image", cropped_img)


# cv2.namedWindow(windowName)
# cv2.setMouseCallback(windowName, crop_image)

# while True:
#     cv2.imshow(windowName, image)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cv2.destroyAllWindows()


# # latihan Crop Image menggunakan mouse event pada click

x_start, y_start, x_end, y_end = 0, 0, 0, 0

windowName = "Original Image"

image = cv2.imread('data1\\tiga_tepung.jpg')


def crop_image(event, x, y, flags, param):

    global x_start, y_start, x_end, y_end
    global croped_img

    # if the left mouse button clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        croped_img = image[y_start:y_end, x_start:x_end]
        cv2.imshow("Cropped Image", croped_img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.imwrite("croped_image.jpg", croped_img)
        print("croped image saved successfully!")


cv2.namedWindow(windowName)
cv2.setMouseCallback(windowName, crop_image)


while True:
    cv2.imshow(windowName, image)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

# # 2. Image Resize
# resize image (new_widht, new_height)
# img_resize = cv2.resize(img, (100, 250))

# # show image
# cv2.imshow('Original Image', img)
# cv2.imshow('Resized Image', img_resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# resize dengan menggunakan rasio ukuran original (hitung manual)
# ratio = float(input("masukan rasio resize [0 - 1.0]: "))
# h, w, c = img.shape

# width = int(w*ratio)
# height = int(h*ratio)

# resize image (new_width, new_height)
# img_resize = cv2.resize(img, (width, height))

# show image
# cv2.imshow('Original Image', img)
# cv2.imshow('Resized Image', img_resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# resize image (new_widht, new_height)
# img_resize = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# show image
# cv2.imshow('Original Image', img)
# cv2.imshow('Resized Image', img_resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------- shringking -------

# resize image (new_widht, new_height)
# img_resize = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# img_resize_INTER_NEAREST = cv2.resize(
#     img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
# img_resize_INTER_AREA = cv2.resize(
#     img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# show image
# cv2.imshow('Original Image', img)
# cv2.imshow('INTER_LINEAR Resized Image', img_resize)
# cv2.imshow('INTER_NEAREST Resized Image', img_resize_INTER_NEAREST)
# cv2.imshow('INTER_AREA Resized Image', img_resize_INTER_AREA)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------- larging -------

# resize image (new_widht, new_height)
# img_resize = cv2.resize(img, (0, 0), fx=2.5, fy=2.5)
# img_resize_INTER_CUBIC = cv2.resize(
#     img, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
# img_resize_INTER_NEAREST = cv2.resize(
#     img, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
# img_resize_INTER_AREA = cv2.resize(
#     img, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_AREA)

# show image
# cv2.imshow('Original Image', img)
# cv2.imshow('INTER_LINEAR Resized Image', img_resize)
# cv2.imshow('INTER_CUBIC Resized Image', img_resize_INTER_CUBIC)
# cv2.imshow('INTER_NEAREST Resized Image', img_resize_INTER_NEAREST)
# cv2.imshow('INTER_AREA Resized Image', img_resize_INTER_AREA)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# -----Resize in window using mouse event click-----

# windowName = "Original Image"
# is_resize = False

# h, w, c = img.shape
# y_end, x_end, last_y_end, last_x_end = h, w, h, w

# background = np.zeros((int(h*1.7), int(w*1.7), c)).astype(np.uint8)
# bg_h, bg_w, bg_c = background.shape


# def resize_image(event, x, y, flags, param):

#     global x_end, y_end, is_resize, bg_h, bg_w

# if the left mouse button clicked
# if event == cv2.EVENT_LBUTTONDOWN:
#     x_end, y_end = x, y
#     is_resize = True

#mouse is moving
# elif event == cv2.EVENT_MOUSEMOVE:
#     x_end, y_end = x, y
# set to max size background if x, y mouse larger than size background
# if x_end > bg_w:
#     x_end = bg_w
# if y_end > bg_h:
#     y_end = bg_h

# if the left mouse button was released
#     elif event == cv2.EVENT_LBUTTONUP:
#         is_resize = False


# cv2.namedWindow(windowName)
# cv2.setMouseCallback(windowName, resize_image)

# while True:
#     template = background.copy()
#     if is_resize:
#         template[:y_end, :x_end] = cv2.resize(img, (x_end, y_end))
#         last_y_end, last_x_end = y_end, x_end
#     else:
#         template[:last_y_end, :last_x_end] = cv2.resize(
#             img, (last_x_end, last_y_end))

#     cv2.imshow(windowName, template)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cv2.destroyAllWindows()

# # 3. Image Blending
# alpha = float(input("Enter alpha [0 - 1.0]: "))

# img1 = cv2.imread()
# img2 = cv2.imread()

# beta = (1.0 - alpha)
# blending_img = cv2.addWeighted(img1, alpha, img2, beta, 0.0)

# cv2.imshow('Blending Result', blending_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# blend image with different size, using cv2.resize() -> strech
# alpha = float(input("Enter alpha [0.0-1.0] : "))

# h, w, c = img.shape

# img1 = cv2.imread('data1\\tepung-tepung.jpeg')
# img1 = cv2.resize(img1, (w, h))

# beta = (1.0 - alpha)
# 0.0 adalah gamma yang disetel ke default, fungsi gamma untuk mengatur kecerahan
# blending_img = cv2.addWeighted(img, alpha, img1, beta, 0.0)

# cv2.imshow("Blending Result", blending_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Blend image with different size, using cv2.resize() -> fit window
# img = cv2.imread('data1\\tiga_tepung.jpg')
# h1, w1, c1 = img1.shape

# img2 = cv2.imread('data1\\tepung-tepung.jpeg')
# h2, w2, c2 = img2.shape

# frame_img = np.ones((h1, w1, c1)).astype(np.uint8)*255

# if h2 > w2:
#     h = h1
#     w = int(w2*h1/h2)
# else:
#     w = w1
#     h = int(h2*w1/w2)

# frame_img[0:h, 0:w] = cv2.resize(img2, (w, h))

# beta = (1.0 - alpha)
# blending_img = cv2.addWeighted(img1, alpha, frame_img, beta, 0.0)

# cv2.imshow('Blending Result', blending_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Menggunakan Trackbar pada OpenCV
# alpha_max_value = 100
# alpha_default_value = 50

# title_window = "Image Blending"

# img1 = cv2.imread('data1\\tepung2.jpeg')
# img2 = cv2.imread('data1\\tepung-tepung.jpeg')


# def on_trackbar(val):
#     alpha = val / alpha_max_value
#     beta = (1.0 - alpha)
#     img_blend = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
#     cv2.imshow(title_window, img_blend)


# cv2.namedWindow(title_window)
# cv2.createTrackbar('alpha', title_window, alpha_default_value,
#                    alpha_max_value, on_trackbar)

# on_trackbar(0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # create overlay
# overlay = np.zeros((25, 75, 3)).astype(np.uint8)
# overlay[:, :, 1] = 255  # set 'B; layer to 255 (color blue)
# h, w, c = overlay.shape

# img_blend = cv2.addWeighted(img[40: 40+h, 40:40+w], 1, overlay, alpha, 0.0)
# img[40: 40+h, 40:40+w] = img_blend

# cv2.imshow("Overlay Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindow()

# # 4. Image Color Conversion

# # convert BGR to Gray
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Original Image', img)
# cv2.imshow('Grayscale Image', img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # convert BGR to RGB
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imshow('Original Image', img)
# cv2.imshow('RGB Image', img_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # convert BGR to BGRA
# img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# cv2.imshow('Original Image', img)
# cv2.imshow('BGRA Image', img_bgra)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(img_bgra.shape)
# print(img_bgra[0, 0])

# img_bgra[:, :, 3] = 205

# cv2.imwrite("coba transparency.png", img_bgra)

# cv2.imwrite("coba transparency.jpg", img_bgra)

# # 5. Buatlah Overlay warna putih dengan alpha input dari trackbar dan size overlay dari mouse event click,
