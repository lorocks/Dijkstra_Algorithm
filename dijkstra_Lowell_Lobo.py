from queue import PriorityQueue
import numpy as np
import math
import cv2
import time
import os
import matplotlib.pyplot as plt



height = 500 # y size
width = 1200 # x size

grid = np.full((height*width), -1)

starting_x = 1
starting_y = 1

visited = PriorityQueue()
open = PriorityQueue()

goal_x = 35
goal_y = 108
goal_index = goal_x + (width * goal_y)


# Add obstacles (change array )

#
current_pos = starting_x + (width * starting_y)
if grid[current_pos] == -1:
  open.put((0, current_pos, -1))
else:
  print("Starting position invalid, obstacle exists")


while not open.empty():
  explore = open.get()
  current_pos = explore[1]

  if not grid[current_pos] == 3:
    grid[current_pos] = 3
    visited.put(explore)
    # print(explore)

    if current_pos == goal_index:
      last_explored = explore
      print("Goal path found")
      break

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

  # add to open and grid only if not obstacle
    for neighbour in neighbours_s:
      if grid[neighbour] == 1 or grid[neighbour] == 3:
        continue
      grid[neighbour] == 2
      open.put((round(explore[0] + 1, 3), neighbour, current_pos))

    for neighbour in neighbours_d:
      if grid[neighbour] == 1 or grid[neighbour] == 3:
        continue
      grid[neighbour] == 2
      open.put((round(explore[0] + 2**0.5, 3), neighbour, current_pos))


backtrack = last_explored
while backtrack[2] != -1:
  grid[backtrack[1]] = 5
  # print(backtrack[1])
  for location in visited.queue:
    if location[1] == backtrack[2] and location[0] < backtrack[0]:
      backtrack = location

data = grid.reshape((height, width))
plt.imshow( data , cmap = 'magma' )