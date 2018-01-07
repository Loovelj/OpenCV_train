import cv2
import numpy as np


def nothing(x):
    pass

# Take each frame
img=cv2.imread('./cut.png')

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


cv2.namedWindow('image')


# create trackbars for color change
cv2.createTrackbar('H1','image',50,255,nothing)
cv2.createTrackbar('S1','image',0,255,nothing)
cv2.createTrackbar('V1','image',0,255,nothing)

# create trackbars for color change
cv2.createTrackbar('H2','image',255,255,nothing)
cv2.createTrackbar('S2','image',255,255,nothing)
cv2.createTrackbar('V2','image',255,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)



while(1):

    # get current positions of four trackbars
    h1 = cv2.getTrackbarPos('H1','image')
    s1 = cv2.getTrackbarPos('S1','image')
    v1 = cv2.getTrackbarPos('V1','image')

    h2 = cv2.getTrackbarPos('H2','image')
    s2 = cv2.getTrackbarPos('S2','image')
    v2 = cv2.getTrackbarPos('V2','image')
  
  
    # define range of blue color in HSV
    lower_blue = np.array([h1,s1,v1])
    upper_blue = np.array([h2,s2,v2])


  
#    # define range of blue color in HSV
#    lower_blue = np.array([90,90,90])
#    upper_blue = np.array([255,255,255])


#    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(img, lower_blue, upper_blue)
    output = cv2.bitwise_not(img, img, mask = mask)
   
    
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        cv2.imshow('image',mask)
    else:
        cv2.imshow('image',output)

        
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
