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


def get_initial_threshold(x,y,image):
    neighbors = get_neighbors(x,y,image)
    neighbors = [image[x[0],x[1]] for x in neighbors]
    return np.median(neighbors)

def level_trace(x,y,image, threshold):
    visited = set()
    visited.add((x,y))
    to_visit = set()
    og_value = image[x,y]
    start = (x,y)
    curr_pixel = (x,y)

    i = 0
    while True:
        neighbors = get_neighbors(curr_pixel[0],curr_pixel[1],image)
        neighbors = [x for x in neighbors if x not in visited]
        to_visit.update([neighbor for neighbor in neighbors if (image[neighbor] < threshold)])
        
        if len(to_visit) == 0:
            break
        else: 
            curr_pixel = to_visit.pop()
            visited.add(curr_pixel)
        i +=1
        
    return visited

def calculate_gradient(x,y,image):
    x_grad = (image[x+1,y] - image[x-1,y])/2
    y_grad = (image[x,y+1] - image[x,y-1])/2
    return np.array([x_grad, y_grad])

def level_trace_2(x,y,image, threshold):
    #x, y are the starting pixels
    #calculate the gradients at the starting pixel
    #move in the direction of the zero gradient
    # add the current pixel to the visited set
    # if the current pixel is the same as the starting pixel, stop
    # if the current pixel is not the same as the starting pixel,
    #   calculate the gradients at the current pixel
    image = np.array(image, dtype=int)

    visited = set()
    visited.add((x,y))
    # to_visit = set()
    curr_value = image[x,y]
    # start_value = curr_value
    start = (x,y)
    curr_pixel = (x,y)
    i = 0

    while True:
        curr_value = image[curr_pixel[0],curr_pixel[1]]
        curr_gradient = calculate_gradient(curr_pixel[0],curr_pixel[1],image)
        neighbors = get_neighbors(curr_pixel[0],curr_pixel[1],image)
        neighbors = np.array([x for x in neighbors if x not in visited])
        all_values = np.array([image[x[0],x[1]] for x in neighbors])
        gradients = np.array([calculate_gradient(neighbor[0],neighbor[1],image) for neighbor in neighbors])

        if len(neighbors) == 0:
            break

        # if np.max(gradients) - np.min(gradients) < 3:
        #     break
        

        #pick the pixel with a gradient most similar to the gradient at the curr pixel
        grad_distances = np.linalg.norm(gradients - curr_gradient, axis=1)
        val_distances = np.linalg.norm(all_values - curr_value)

        # pdb.set_trace()
        distances = grad_distances + val_distances
        # pdb.set_trace()
        min_index = np.argmin(distances)
        curr_pixel = neighbors[min_index]

        visited.add((curr_pixel[0],curr_pixel[1]))
    
        if curr_pixel[0] == start[0] and curr_pixel[1] == start[1]:
            break
        i += 1

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

        visited = level_trace_2(y,x,img,thresh)

        # displaying the output image over the original image   
        for i in visited:     
            img[i[0],i[1]] = 255
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
