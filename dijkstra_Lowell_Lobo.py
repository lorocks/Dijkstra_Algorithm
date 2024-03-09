from queue import PriorityQueue
import numpy as np
import math
import cv2
import time
import os



height = 500 # y size
width = 1200 # x size

timestep = 0


open = PriorityQueue()

grid = np.full((height*width), -1)
backtrack_grid = grid.copy()


valid = False
while not valid:
  starting_x = int(input("\nEnter starting x position:"))
  starting_y = int(input("\nEnter starting y position:"))

  current_pos = starting_x + (width * starting_y)
  try:
    if grid[current_pos] == -1:
      grid[current_pos] = 0
      backtrack_grid[current_pos] = -1
      open.put((0, current_pos))
      valid = True
    else:
      print("\nStarting position invalid, obstacle exists, Enter again\n")
  except:
    print("\nStarting position invalid, obstacle exists, Enter again\n")

valid = False
while not valid:
  goal_x = int(input("\nEnter goal x position:"))
  goal_y = int(input("\nEnter goal y position:"))
  goal_index = goal_x + (width * goal_y)

  try:
    if grid[goal_index] == -1:
      valid = True
    else:
      print("\nGoal position invalid, obstacle exists, Enter again\n")
  except:
    print("\nGoal position invalid, obstacle exists, Enter again\n")

while not open.empty():
  explore = open.get()
  current_pos = explore[1]

  if not grid[current_pos] == -13:
    timestep += 1
    grid[current_pos] = -13
    # print(explore)

    if current_pos == goal_index:
      last_explored = explore
      print("\nGoal path found")


    x_pos = int(current_pos % width)
    y_pos = int((current_pos - (current_pos % width))/width)
    neighbours_s = []
    neighbours_d = []

    if x_pos > 0: # left action
      neighbours_s.append(current_pos - 1)
    if x_pos < width - 1: # right action
      neighbours_s.append(current_pos + 1)
    if y_pos > 0: # up action
      neighbours_s.append(((y_pos - 1)*width) + x_pos)
    if y_pos < height - 1: # down action
      neighbours_s.append(((y_pos + 1)*width) + x_pos)
    if x_pos > 0 and y_pos > 0: # left up action
      neighbours_d.append(((y_pos - 1)*width) + (x_pos - 1))
    if x_pos < width - 1 and y_pos > 0: # right up action
      neighbours_d.append(((y_pos - 1)*width) + (x_pos + 1))
    if x_pos > 0 and y_pos < height - 1: # left down action
      neighbours_d.append(((y_pos + 1)*width) + (x_pos - 1))
    if x_pos < width - 1 and y_pos < height - 1: # right down action
      neighbours_d.append(((y_pos + 1)*width) + (x_pos + 1))

  # add to open and grid only if not obstacle, RECHECK THIS CRAP
    for neighbour in neighbours_s:
      if grid[neighbour] < -10:
        continue
      cost = explore[0] + 1
      if grid[neighbour] == -1 or grid[neighbour] > cost:
        grid[neighbour] = cost
        backtrack_grid[neighbour] = current_pos
        open.put((cost, neighbour))

    for neighbour in neighbours_d:
      if grid[neighbour] < -10:
        continue
      cost = explore[0] + 2**0.5
      if grid[neighbour] == -1 or grid[neighbour] > cost:
        grid[neighbour] = cost
        backtrack_grid[neighbour] = current_pos
        open.put((cost, neighbour))


index = goal_index

while backtrack_grid[index] > 0:
  grid[index] = -4
  index = backtrack_grid[index]


# Image show
data = np.copy(grid)
# data[data == -1] = 60
# data[data == 2] = 100
# data[data == 3] = 140
data = data.reshape((height, width))
image = np.zeros((data.shape[0], data.shape[1], 3))
image[data == -1] = (224, 224, 224)
image[data == -11] = (0, 0, 0)
image[data == -12] = (125, 125, 125)
image[data == -13] = (152, 251, 152)
image[data == -4] = (0, 0, 255)
image[data >= 0] = (255, 0, 0)

image = cv2.flip(image, 0)
cv2.imshow("Frame", image)