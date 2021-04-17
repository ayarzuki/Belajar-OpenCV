import cv2
import numpy as np

# # Vehicle Counting
car_cascade = cv2.CascadeClassifier('haarcascade\\cars.xml')

cap = cv2.VideoCapture("pertemuan14\\cars_video.mp4")

count = 0
prev_y = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    h, w, c = frame.shape
    x1, y1, x2, y2 = int(w*0.1), int(h*0.8), int(w*0.9), int(h*0.8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, minNeighbors=5)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        cy = (y+h//2)
        if (y1 - 30) < cy and (y1 + 30) > cy and abs(prev_y - cy) > 20:
            count += 1
            prev_y = cy

    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(frame, "Vehicle Count : %d" % count, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Detect Cars', frame)

    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()

# # Vehicle and Direction Detecting Cars
# car_cascade = cv2.CascadeClassifier('haarcascade\\cars.xml')

# cap = cv2.VideoCapture("pertemuan14\\cars_video.mp4")

# count = 0
# prev_y = 0
# direction = ""
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#     h, w, c = frame.shape
#     x1, y1, x2, y2 = int(w*0.1), int(h*0.7), int(w*0.9), int(h*0.7)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cars = car_cascade.detectMultiScale(gray, minNeighbors=5)
#     for (x, y, w, h) in cars:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
#         cy = (y+h//2)
#         if (y1 - 30) < cy and (y1 + 30) > cy and abs(prev_y - cy) > 20:
#             count += 1
#             if prev_y > cy:
#                 direction = "up"
#             else:
#                 direction = "down"
#             prev_y = cy

#     cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
#     cv2.putText(frame, "Vehicle Count : %d | Direction : %s" % (count, direction),
#                 (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
#     cv2.imshow('Detect Cars', frame)

#     if cv2.waitKey(1) == ord('q'):
#         break
# cv2.destroyAllWindows()
