import cv2

cap = cv2.VideoCapture(0)

_, frame = cap.read()

cv2.imshow('Nope', frame)
cv2.waitKey(0)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('Nope', frame)
cv2.waitKey(0)