import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing

def gammaCorrection(image, g=1):
    ig = 1 / g

    lut = (np.arange(256) / 255) ** ig * 255
    lut = lut.astype("uint8")
    return cv2.LUT(image, lut)

cap = cv2.VideoCapture("balls/videos/balls.mp4")
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)


while cap.isOpened():
    _, frame = cap.read()
    
    if _:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray = gammaCorrection(gray, 1.25)
        
        _, thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        mask = np.ones((6, 6))
        
        thresh = cv2.erode(thresh, mask, iterations=4)
        thresh = cv2.dilate(thresh, mask, iterations=2)
        labeled = label(thresh)
        
        print(labeled.max())
        
        cv2.putText(frame, f"Count = {labeled.max()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    
        cv2.imshow('Camera', frame)
        cv2.imshow('Thresh', thresh)
    key = cv2.waitKey(15)
    if key == ord('q'):
        break
    
    
    
cap.release()
cv2.destroyAllWindows()