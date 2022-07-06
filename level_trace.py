# importing the module
from email.mime import image
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

def level_trace(x,y,image, threshold):
    #check if this is a uniform area
    #if it is, return the starting pixel
    neighbors = get_neighbors(x,y,image)
    neighbors = [image[x[0],x[1]] for x in neighbors]
    if np.max(neighbors) - np.min(neighbors) < 10:
        return [(x,y)]

    #get the gradiets of 

    # x, y are the starting pixels

    start = (x,y)

    image = np.array(image, dtype=int)

    visited = set()

    # #apply the threshold to the image
    image[image < threshold] = 0
    image[image >= threshold] = 255

    start_x = x
    start_y = y

    curr_x = x
    curr_y = y
    
    start_mean = get_mean_val(x,y,image)

    i = 0

    while True: 
        neighbors = get_neighbors(curr_x,curr_y,image)
        neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
        if len(neighbors) == 0:
            break
        neighbor_means = [get_mean_val(neighbor[0],neighbor[1],image) for neighbor in neighbors]
        neighbor_means = np.array(neighbor_means)
        diffs = neighbor_means - start_mean
        diffs = np.abs(diffs)
        min_index = np.argmin(diffs)
        visited.add((curr_x,curr_y))

        curr_x = neighbors[min_index][0]
        curr_y = neighbors[min_index][1]

        if i == 10:
            visited.remove((start_x,start_y))

        if curr_x == start_x and curr_y == start_y:
            break
        i += 1

    return visited

def calculate_gradient(x,y,image):
    x_grad = (image[x+1,y] - image[x-1,y])/2
    y_grad = (image[x,y+1] - image[x,y-1])/2
    return np.array([x_grad, y_grad])

def level_trace_2(x,y,image, threshold):
    #check if this is a uniform area
    #if it is, return the starting pixel
    neighbors = get_neighbors(x,y,image)
    neighbors = [image[x[0],x[1]] for x in neighbors]
    if np.max(neighbors) - np.min(neighbors) < 10:
        return [(x,y)]
    
    start = (x,y)

    image = np.array(image, dtype=int)

    visited = set()

    # #apply the threshold to the image
    image[image < threshold] = 0
    image[image >= threshold] = 255


    #make sure to check if the current pixel has a value of 255
    #if it doesn't, change curr_x and curr_y to be a neighbor of the current pixel
    if image[x,y] == 0:
        neighbors = get_neighbors(x,y,image)
        neighbors_vals = [image[x[0],x[1]] for x in neighbors]
        min_index = np.argmin(neighbors_vals)
        x = neighbors[min_index][0]
        y = neighbors[min_index][1]

    to_visit = []
    while True: 
        neighbors = get_neighbors(x,y,image)
        neighbors = [neighbor for neighbor in neighbors if neighbor not in visited and image[neighbor] == 255]
        if len(neighbors) == 0:
            break
        neighbor_vals = [image[x[0],x[1]] for x in neighbors]

        neighbor_vals = np.array(neighbor_vals)
        diffs = neighbor_vals - image[x,y]
        diffs = np.abs(diffs)
        min_index = np.argmin(diffs)
        visited.add((x,y))

        to_visit.append((neighbors[min_index][0],neighbors[min_index][1]))

        next_thing = to_visit.pop(0)

        x = next_thing[0]
        y = next_thing[1]




    return visited



# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
    # if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDOWN:

        img = cv2.GaussianBlur(cv2.imread('double-scan.jpg',0), (5,5), 0)
		# displaying the coordinates
		# on the Shell
        print(x, ' ', y)

        thresh = get_initial_threshold(y,x,img)

        # img[img < thresh] = 0
        # img[img >= thresh] = 255

        # print(img[x,y])

        visited = level_trace(y,x,img,thresh)

        # # displaying the output image over the original image   
        for i in visited:
            # pdb.set_trace()
            img[i] = 255

        cv2.imshow('image', img)





# driver function
if __name__=="__main__":

	# reading the image
    img = cv2.imread('double-scan.jpg',0)
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
