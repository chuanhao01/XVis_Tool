import cv2
import numpy as np

mask = np.zeros((400, 400, 3), dtype='uint8')
# print(mask.shape)

img2d = np.zeros((100, 100), dtype='uint8')

img2d[:50, :50] = 255
img3d = []
for c_channel in range(3):
    img3d.append(img2d)

print(img3d)
img3d = np.array(img3d)
img3d = np.transpose(img3d)
print(img3d.shape)

mask[50:150, 50:150, :3] = img3d 

cv2.imshow('hello', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
