import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Read Image
# img = cv2.imread("data1\\tiga_tepung.jpg")
img = mpimg.imread("data1\\tiga_tepung.jpg")
# print(type(img))
# print(img.shape)
# (B, G, R) = img[0, 0]
# print("R=%d, G=%d, B=%d" % (R, G, B))

# 2. Show Image
# res = cv2.imshow('Uji Gambar Tepung', img)
# cv2.waitKey(0)
# res = cv2.waitKey(4000)
# print('You pressed : %s' % chr(res) if res >=
#       0 and res <= 127 else '<unknown>')
# cv2.destroyAllWindows()

# 3. Menggunakan Matplotlib
# img_gray = img[:, :, 0]
# plt.imshow(img_gray, cmap='gray')
# plt.show()

# 4. Convert menggunakan cv2.cvtColor
# (Convert BGR to RGB Color)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# plt.imshow(img_rgb)
# plt.imshow(img)
# plt.show()

# 5. Two GUI image using OpenCV GUI
# img1 = cv2.imread('data1\\dua_tepung.jpeg')
# img2 = cv2.imread('data1\\tepung1.jpeg')
# cv2.imshow('myapp 1', img1)
# cv2.imshow('myapp 2', img2)
# cv2.waitKey(0)  # display the window infinitely until any keypress
# cv2.destroyAllWindows()

# 6. Play Video
# load video
# cap = cv2.VideoCapture('data1\\mojokerto.mp4')

# # iterate for each frame in video
# while cap.isOpened():
#     # get image on each frame
#     ret, frame = cap.read()
#     if ret == True:
#         # show image
#         cv2.imshow('Frame', frame)
#         # wait 25ms per frame and close using 'q'
#         if cv2.waitKey(25) == ord('q'):
#             break
# # close video
# cap.release()
# # close window
# cv2.destroyAllWindows()

# 7. Capture Photo
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()

# if ret:
#     cv2.imwrite("data1\\coba_capture1.jpg", frame)
# else:
#     print("can't save photo")

# # close video
# cap.release()

# 8. Capture Video
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Define the codec and create VideoWriter object
# fourcc is a 4-byte code used to specify the video codec
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # fourcc for .avi

out = cv2.VideoWriter('mojokerto.mp4', fourcc, 20,
                      (320, 240))  # name, fourcc, fps, size

cv2.VideoWriter
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite("foro_from_video.jpg", frame)
            print("foto captured!")
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
