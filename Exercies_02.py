import cv2
import numpy as np

def process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    minVal = 100
    maxVal = 200
    image = np.where(image <= minVal, minVal, image)
    image = np.where(image >= maxVal, maxVal, image)
    return image

myimage = cv2.imread("test.jpg")
myimage = process(myimage)



image = np.zeros((480, 640, 3), dtype="uint8")
#image2 = np.zeros((480, 640, 3), dtype="unit8")
print("Hello")

a = [0, 10, 20, 30, 40, 50]
print(a[1:3])
image[:100,:] = (0,0,255)
#image2[:200,:]  = (255,0,0)
image[100:200,:] = 128
#image[:,:,0:,1] = 128

myimage = cv2.resize(myimage, dsize=(0,0), fx=0.1, fy=0.1)
myimage = cv2.resize(myimage, dsize=(0,0), fx=10.0, fy=10.0, interpolation=cv2.INTER_NEAREST)

cv2.imshow("Image", image)
cv2.imshow("Image2", myimage);
cv2.waitKey(-1)
cv2.destroyAllWindows()


# import tensorflow as tf

# print(tf.__version__)