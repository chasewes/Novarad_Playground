import numpy as np
import matplotlib.pyplot as plt
from fibheap import *
import pdb
from tqdm import tqdm
import heapq
from heapdict import *

image = plt.imread('shapes.png',)

def get_neighbors(pixel, ymax, xmax):
    location = pixel[0]
    x = location[0]
    y = location[1]

    tl = (x-1,y-1)
    tm = (x, y-1)
    tr = (x+1, y-1)
    ml = (x-1,y)
    mr = (x+1, y)
    bl = (x-1,y+1)
    bm = (x, y+1)
    br = (x+1, y+1)
    neighbors = [tl, tm, tr, ml, mr, bl, bm, br]
    # out = []

    return [neighbor for neighbor in neighbors if not (neighbor[1] >= xmax or neighbor[0] >= ymax or neighbor[0] < 0 or neighbor[1] < 0)]         
         
labCRT = np.zeros(image.shape[:-1])
distCRT = np.full(image.shape[:-1], np.inf)

labCRT[400,400] = 1
distCRT[400,400] = 0

labCRT[100,100] = 2
distCRT[100,100] = 0

ymax = len(image) #num rows
xmax = len(image[0]) #num columns

# heap = {(i,j): distCRT[i,j] for i in range(len(image)) for j in range(len(image[0]))}

heap = heapdict()

for j in range(len(image[0])):
    for i in range(len(image)):
        heap[(i,j)] = distCRT[i,j]

while len(heap):
    smallest_pixel = heap.popitem()

    neighbors = get_neighbors(smallest_pixel, ymax, xmax)
    for neighbor in neighbors:

        #calculate the distance from the neighbor to the pixel
        dist = smallest_pixel[1] + np.linalg.norm(image[neighbor] - image[smallest_pixel[0]])

        if dist < distCRT[neighbor]:
            labCRT[neighbor] = labCRT[smallest_pixel[0]]
            heap[neighbor] = dist
            distCRT[neighbor] = dist

