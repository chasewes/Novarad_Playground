# importing the module
from email.mime import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb



# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params,img):

    if event == cv2.EVENT_LBUTTONDOWN:

    # if event == cv2.EVENT_RBUTTONDOWN and event == cv2.EVENT_MOUSEMOVE:

        print(x, ' ', y)

        img[y,x] = 255




        cv2.imshow('image', img)

    # elif event == cv2.EVENT_MOUSEMOVE:

    #     img = cv2.imread('test.png',0)
	# 	# displaying the coordinates
	# 	# on the Shell
    #     print(x, ' ', y)

    #     thresh = get_initial_threshold(y,x,img)

    #     visited = level_trace(y,x,img,thresh)

    #     # displaying the output image over the original image   
    #     for i in visited:     
    #         img[i[0],i[1]] = 255
    #     cv2.imshow('image', img)


# driver function
if __name__=="__main__":

	# reading the image
    img = cv2.imread('double-scan.jpg',0)
    
	# displaying the image
    cv2.imshow('image', img)

        # setting mouse handler for the image
        # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

        # wait for a key to be pressed to exit
    cv2.waitKey(0)

        # close the window
    cv2.destroyAllWindows()
