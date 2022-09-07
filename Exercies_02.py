import cv2
import numpy as np

image = np.zeros((480, 640, 3), dtype="uint8")
image2 = np.zeros((480, 640, 3), dtype="unit8")
print("Hello")

a = [0, 10, 20, 30, 40, 50]
print(a[1:3])
image[:100,:] = (0,0,255)
image2[:200,:]  = (255,0,0)

cv2.imshow("Image", image)
cv2.waitKey(-1)
cv2.destroyAllWindows()


# import tensorflow as tf

# print(tf.__version__)