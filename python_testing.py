import cv2

cap = cv2.VideoCapture(0)

while True:
    ret_run, frame = cap.read()
    print(frame.shape)
    cv2.imshow('Test', frame)
    cv2.waitKey(100)