import numpy as np 
import cv2
import sys






cap = cv2.VideoCapture("10k20v.avi")

tracker = cv2.TrackerMIL_create()

success, img = cap.read()

bbox = cv2.selectROI('frame',img,False)



tracker.init(img,bbox)

def drawBox(img, bbox):
    x , y , w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img, "Matt's tracking",(75,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,1)

while(True):
    
    success, bbox = tracker.update(img,) 
    success, img = cap.read()
        
    if success:
         drawBox(img, bbox)
         print(bbox)
    else:
        cv2.putText(img, "track lost",(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,1)

    
    
   # ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',img)
    drawBox(img, bbox)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
