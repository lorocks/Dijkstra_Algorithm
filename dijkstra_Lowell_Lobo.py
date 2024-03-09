from queue import PriorityQueue
import numpy as np
import math
import cv2
import time
import os


def createGrid(height, width, bounding_location, padding = 0, wall = False, wall_padding = 0):
  image = np.full((height, width, 3), 255, dtype=np.uint8)

  image = setObstaclesAndTruePadding(image, bounding_location, padding)

  if wall:
    image = setWallPadding(image, wall_padding)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  grid = np.full((height, width), -1)
  grid[gray == 125] = -12
  grid[gray == 0] = -11
  grid = grid.reshape(-1)

  return grid

def setWallPadding(image, padding):
  height, width, _ = image.shape

  points = [
      (0, 0), (padding, height),
      (0, 0), (width, padding),
      (0, height - padding), (width, height),
      (width - padding, 0), (width, height),
  ]

  for i in range(0, len(points), 2):
    cv2.rectangle(image, points[i], points[i+1], (125, 125, 125), -1)

  return image

def setObstaclesAndTruePadding(image, bounding_location, padding):
  if padding > 0:
    doPadding = True
    paddings = [
        [padding, padding],
                [-padding, -padding],
     [padding, -padding],
      [-padding, padding],
                [0, padding],
                [0, -padding],
                [padding, 0],
                [-padding, 0]
                ]

  for obstacle in bounding_location:

    if len(obstacle) == 2:
      if doPadding:
        for pad in paddings:
          cv2.rectangle(image, (obstacle[0][0] - pad[0], obstacle[0][1] - pad[1]), (obstacle[1][0] + pad[0], obstacle[1][1] + pad[1]), (125, 125, 125), -1)
      cv2.rectangle(image, obstacle[0], obstacle[1], (0, 0, 0), -1)
    else:
      arr = np.array(obstacle, dtype=np.int32)
      if doPadding:
        for pad in paddings:
          length = len(obstacle)
          arrr = np.full((length, 2), pad)
          cv2.fillPoly(image, pts=[np.subtract(arr, arrr)], color=(125, 125, 125))
      cv2.fillPoly(image, pts=[arr], color=(0, 0, 0))

  return image


start = time.time()
height = 500 # y size
width = 1200 # x size
padding = 5

timestep = 0

obstacle_file_path = ""

obstacle_bounding_boxes = [
    [[175, 100], [100, 500] ], [[275, 400], [350, 0]],
    [[650 - (75*(3**0.5)), 325], [650 - (75*(3**0.5)), 175], [650, 100], [650 + (75*(3**0.5)), 175], [650 + (75*(3**0.5)), 325], [650, 400]],
    [[900, 450], [1100, 450], [1100, 50], [900, 50], [900, 125], [1020, 125], [1020, 375], [900, 375]],
                            ]

open = PriorityQueue()



if os.path.exists(obstacle_file_path) and os.path.isfile(obstacle_file_path):
  pass
else:
  # Enter array manually maybe through prompt
  grid = createGrid(height, width, obstacle_bounding_boxes, padding, True, padding)
  backtrack_grid = np.full((height*width), -1)


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


timef = time.time() - start
print(f"Goal found in {math.floor(timef/60)} minutes and {(timef % 60):.2f} seconds")

start = time.time()
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
image = np.uint8(image)

cv2.imshow(image)

timef = time.time() - start
print(f"Backtracking done in {math.floor(timef/60)} minutes and {(timef % 60):.2f} seconds")
