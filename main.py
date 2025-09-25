import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # height
segmentor = SelfiSegmentation(1)

imgBG = cv2.imread("images/VirtualBG.png")

# Resize the background image to match the webcam frame size
imgBG = cv2.resize(imgBG, (640, 480))

while True:
  success, img = cap.read()
  img = cv2.flip(img, 1) # Flip user's webcam view only
  imageOut = segmentor.removeBG(img, imgBG, 0.53)
  cv2.imshow("Webcam", imageOut)
  if cv2.waitKey(25) & 0xFF == ord("q"): # press "q" to quit the webcam/window
    break

cap.release()
cv2.destroyAllWindows()