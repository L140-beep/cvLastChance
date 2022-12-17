import cv2
import numpy as np
import time


def get_center(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)

        if radius > 10:
            return x, y, radius
    
    return None


#100, 173, 127
blue_lower = (70, 143, 97)
blue_upper = (110, 203, 157)

cam = cv2.VideoCapture(0)

cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cam.set(cv2.CAP_PROP_EXPOSURE, 3000)


#v=s/Δt

prev_pos = [0, 0]

prev_time = time.perf_counter() 

while cam.isOpened():
    _, frame = cam.read()
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    blue = get_center(hsv, blue_lower, blue_upper)
    
    if blue is not None:
        x = int(blue[0])
        y = int(blue[1])
        r = int(blue[2])
        cv2.circle(frame, (x, y), r,(255, 0, 0), 3)

        s = ((prev_pos[0] - x) ** 2 + (prev_pos[1] - y) ** 2) ** 0.5 
        
        
        prev_pos = (x, y)
        
        current_time =  time.perf_counter()
        
        delta_t = current_time - prev_time
        
        prev_time = current_time
        
        metres_s = (s * 0.2636) / 1000
        
        v = round(metres_s / delta_t, 3)
        
        cv2.putText(frame, f"Speed: {v} m/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
        
        
        
        
        
        
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
    cv2.imshow('frame', frame)
    

cam.release()
cv2.destroyAllWindows()
