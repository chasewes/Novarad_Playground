# importing the module
from email.mime import image
from tabnanny import check
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
from pyrsistent import s

from sklearn import neighbors

def get_neighbors(x,y,image):
    neighbors = []
    xmin = 0
    xmax = len(image)
    ymin = 0
    ymax = len(image[0])

    if x > 0:
        neighbors.append((x-1,y))
    if x < xmax-1:
        neighbors.append((x+1,y))
    if y > 0:
        neighbors.append((x,y-1))
    if y < ymax-1:
        neighbors.append((x,y+1))
    if x > 0 and y > 0:
        neighbors.append((x-1,y-1))
    if x > 0 and y < ymax-1:
        neighbors.append((x-1,y+1))
    if x < xmax-1 and y > 0:
        neighbors.append((x+1,y-1))
    if x < xmax-1 and y < ymax-1:
        neighbors.append((x+1,y+1))
        
    return neighbors


def get_initial_threshold(x,y,image):
    neighbors = get_neighbors(x,y,image)
    neighbors = [image[x[0],x[1]] for x in neighbors]
    return np.median(neighbors)

def get_mean_val(x,y,image):
    neighbors = get_neighbors(x,y,image)
    neighbors = [image[x[0],x[1]] for x in neighbors]
    return np.mean(neighbors)

def calculate_gradient(x,y,image):
    x_grad = (image[x+1,y] - image[x-1,y])/2
    y_grad = (image[x,y+1] - image[x,y-1])/2
    return np.array([x_grad, y_grad])

def get_neighbors_array(x,y,image):
    neighbors = []
    xmin = 0
    xmax = len(image)
    ymin = 0
    ymax = len(image[0])
    neighbors = np.full((3,3),-1)

    if x > 0 and y > 0:
        neighbors[0,0] = image[x-1,y-1]
    if x > 0 and y < ymax-1:
        neighbors[0,2] = image[x-1,y+1]
    if x < xmax-1 and y > 0:
        neighbors[2,0] = image[x+1,y-1]
    if x < xmax-1 and y < ymax-1:
        neighbors[2,2] = image[x+1,y+1]
    if x > 0:
        neighbors[0,1] = image[x-1,y]
    if x < xmax-1:
        neighbors[2,1] = image[x+1,y]
    if y > 0:
        neighbors[1,0] = image[x,y-1]
    if y < ymax-1:
        neighbors[1,2] = image[x,y+1]
    return neighbors

def check_if_clockwise(points):
    sum = 0
    # for each pair of points in the visited set (connecting the ending point to the starting point),
    # add (x2-x1) * (y2+y1) to the sum
    for i in range(len(points)-1):
        sum += (points[i+1][0] + points[i][0]) * (points[i+1][1] - points[i][1])
    # if the sum is positive, the points are clockwise
    return sum > 0

def level_trace(x,y,img, threshold):
    #check if this is a uniform area
    #if it is, return the starting pixel
    #threshold the image
    image = img.copy()
    image[image < threshold] = 0
    image[image >= threshold] = 255


    neighbors = get_neighbors(x,y,image)
    # print((x,y))
    # print(neighbors)
    neighbors = [image[x[0],x[1]] for x in neighbors]
    # print(neighbors)
    if np.max(neighbors) == 0 or np.min(neighbors) == 255:
        return [(x,y)]

    start = (x,y)

    image = np.array(image, dtype=int)

    visited = []
    #from the starting pixel, perform a radial sweep (moore neighborhood tracing), and add the pixels to the visited set

    seen_blank = False
    curr = start
    prev = 0
    iters = 0
    # continue until we have returned to the starting pixel or we time out 
    while True:
        iters += 1
        #move clockwise around the neighbors 
        #2 3 4
        #1 x 5
        #0 7 6
        #starting at 1, look for the first pixel that is not visited that is above the threshold
        #if you find one, add it to the visited set and continue the radial sweep
        i = prev
        # print("outer")
        while True:
            
            i = (i+1)%8
            if i == 0: 
                neighbor = (curr[0]-1,curr[1]-1)
            elif i == 1:
                neighbor = (curr[0]-1,curr[1])
            elif i == 2:
                neighbor = (curr[0]-1,curr[1]+1)
            elif i == 3:
                neighbor = (curr[0],curr[1]+1)
            elif i == 4:
                neighbor = (curr[0]+1,curr[1]+1)
            elif i == 5:
                neighbor = (curr[0]+1,curr[1])
            elif i == 6:
                neighbor = (curr[0]+1,curr[1]-1)
            else:
                neighbor = (curr[0],curr[1]-1)

            # print(f"{i}: neighbor: {neighbor}, val: {image[neighbor]}, curr: {curr}, curr_val: {image[curr]}, start: {x},{y}")
            if image[neighbor] < threshold:
                seen_blank = True
            if image[neighbor] >= threshold and seen_blank:
                visited.append(curr)
                prev = (i - 4) % 8 
                # prev = i
                curr = neighbor
                seen_blank = False
                break
        if curr == start:
            print("returned to beginning")
            break
        if iters > 10000:
            print("timed out")
            # pdb.set_trace()
            break

    if check_if_clockwise(list(visited)):
        print("clockwise")
    else:
        print("counterclockwise")
    # print(len(visited))
    return visited

def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_MOUSEMOVE:
        img = cv2.imread('override.jpg',0)
		# displaying the coordinates
		# on the Shell
        print(y, ' ', x)

        # thresh = get_initial_threshold(y,x,img)
        thresh = img[y,x]

        visited = level_trace(y,x,img,thresh)

        # # displaying the output image over the original image   
        for i in visited:
            # pdb.set_trace()
            img[i] = 255
        

        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        img = cv2.imread('double-scan.jpg')
        cv2.imwrite('override.jpg',img)
        cv2.imshow('image', img)

    if event == cv2.EVENT_LBUTTONDOWN:
        img = cv2.imread('override.jpg',0)

        thresh = img[y,x]
        contour = np.array(level_trace(y,x,img,thresh))
        x_min = np.min(contour[:,0])
        x_max = np.max(contour[:,0])
        y_min = np.min(contour[:,1])
        y_max = np.max(contour[:,1])

        for y in range(y_min,y_max):
            for x in range(x_min,x_max):
                if cv2.pointPolygonTest(contour, (x,y), False) >= 0:
                    img[x,y] = img[x,y] * -1
        cv2.imwrite('override.jpg',img)
        cv2.imshow('image', img)

# driver function
if __name__=="__main__":

	# reading the image
    img = cv2.imread('Moon Lit v 2  copy.png',0)
    #save the original image to a file called "override.jpg"
    cv2.imwrite('override.jpg',img)

    img = cv2.GaussianBlur(img, (5,5), 0)
    
	# displaying the image
    cv2.imshow('image', img)

        # setting mouse handler for the image
        # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

        # wait for a key to be pressed to exit
    cv2.waitKey(0)

        # close the window
    cv2.destroyAllWindows()
