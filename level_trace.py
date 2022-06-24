# importing the module
from email.mime import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb

def get_neighbors(i,j,image):
    neighbors = []
    xmin = 0
    xmax = len(image[0])
    ymin = 0
    ymax = len(image)

    if i > 0:
        neighbors.append((i-1,j))
    if i < xmax-1:
        neighbors.append((i+1,j))
    if j > 0:
        neighbors.append((i,j-1))
    if j < ymax-1:
        neighbors.append((i,j+1))
    if i > 0 and j > 0:
        neighbors.append((i-1,j-1))
    if i > 0 and j < ymax-1:
        neighbors.append((i-1,j+1))
    if i < xmax-1 and j > 0:
        neighbors.append((i+1,j-1))
    if i < xmax-1 and j < ymax-1:
        neighbors.append((i+1,j+1))
        
    return neighbors


def level_trace(x,y,image):
    visited = set()
    visited.add((x,y))
    # pdb.set_trace()
    #do an isometric level trace
    #always go to the neighbor that is most similar to the current pixel
    #discard the other neighbors
    og_value = image[x,y]
    start = (x,y)
    curr_pixel = (x,y)
    i = 0
    while True:
        neighbors = get_neighbors(curr_pixel[0],curr_pixel[1],image)
        neighbors = [x for x in neighbors if x not in visited]
        #get the neighbor that is most similar to the current pixel
        neighbors = sorted(neighbors, key=lambda x: np.abs(image[x[0],x[1]] - og_value))
        # try:
        if len(neighbors) > 0:
            curr_pixel = neighbors[0]
            visited.add(curr_pixel)
            i += 1
        else:
            break

        visited.add(curr_pixel)
        if i == 1:
            visited.remove(start)
        i +=1
        if curr_pixel == start:
            break
    return visited

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDOWN:


        img = cv2.imread('test.png',0)
		# displaying the coordinates
		# on the Shell
        print(x, ' ', y)
        # pdb.set_trace()
        visited = level_trace(y,x,img)

        # displaying the output image over the original image   
        for i in visited:     
            img[i[0],i[1]] = 255
        cv2.imshow('image', img)


# driver function
if __name__=="__main__":

	# reading the image
    img = cv2.imread('test.png',0)
    
	# displaying the image
    cv2.imshow('image', img)

        # setting mouse handler for the image
        # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

        # wait for a key to be pressed to exit
    cv2.waitKey(0)

        # close the window
    cv2.destroyAllWindows()
