# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from digit import*

# Load the classifier
clf = joblib.load("digits_cls2.pkl")

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame2=frame.copy()
    
    
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_gray.copy(), 120, 255, cv2.THRESH_BINARY_INV)
    # Find contours in the binary image 'im_th'
    _, contours0, hierarchy  = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Rectangular bounding box around each number/contour
    rects = [cv2.boundingRect(ctr) for ctr in contours0]
    # Draw the bounding box around the numbers
    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 255), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
   
    img_counter = 1
    k = cv2.waitKey(1)
    if k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame2)
        print("{} written!".format(img_name))
        
        w1=capture(img_name)
        img_counter += 1
        n1= svm(w1)
        img_name = "Num_{}.png".format(n1)
        cv2.imwrite(img_name, frame2)
        img_counter =img_counter+1
        print(n1)
        
        

    # Resize the image
    if roi.any():
        roi = cv2.resize(roi, (28, 28), frame, interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
    
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('Threshold', im_th)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()