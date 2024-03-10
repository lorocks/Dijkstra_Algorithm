from queue import PriorityQueue
import numpy as np
import math
import cv2
import time
import os


# Function to create a grid for the algorithm
def createGrid(height, width, bounding_location, padding = 0, ptype = "true", wall = False, wall_padding = 0):
  # Create image to add obstacles and padding
  image = np.full((height, width, 3), 255, dtype=np.uint8)

  # Select between obstacle and padding types
  if ptype == "true":
    image = setObstaclesAndTruePadding(image, bounding_location, padding)
  else:
    image = setObstaclesAndCircularPadding(image, bounding_location, padding)

  # Set wall padding
  if wall:
    image = setWallPadding(image, wall_padding)
  
  # Convert 3D array to 2D array
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Create array for algorithm
  grid = np.full((height, width), -1)
  grid[gray == 125] = -12
  grid[gray == 0] = -11
  grid = grid.reshape(-1)

  return grid


# Add wall paddings
def setWallPadding(image, padding):
  height, width, _ = image.shape

  # Define points for wall padding
  points = [
      (0, 0), (padding, height),
      (0, 0), (width, padding),
      (0, height - padding), (width, height),
      (width - padding, 0), (width, height),
  ]

  # Draw padding on image
  for i in range(0, len(points), 2):
    cv2.rectangle(image, points[i], points[i+1], (125, 125, 125), -1)

  return image


# Set obstacles and true padding
def setObstaclesAndTruePadding(image, bounding_location, padding):
  # Set padding if padding is greater than 0
  if padding > 0:
    doPadding = True
    # Define a padding array
    paddings = [
        [padding, padding],
        [-padding, -padding],
        [padding, -padding],
        [-padding, padding],
        [0, padding],
        [0, -padding],
        [padding, 0],
        [-padding, 0],
                ]

  for obstacle in bounding_location:
    if len(obstacle) == 2:
      # Draw paddings
      if doPadding:
        for pad in paddings:
          cv2.rectangle(image, (obstacle[0][0] - pad[0], obstacle[0][1] - pad[1]), (obstacle[1][0] + pad[0], obstacle[1][1] + pad[1]), (125, 125, 125), -1)
      # Draw obstacle
      cv2.rectangle(image, obstacle[0], obstacle[1], (0, 0, 0), -1)
    else:
      arr = np.array(obstacle, dtype=np.int32)
      # Draw paddings
      if doPadding:
        for pad in paddings:
          length = len(obstacle)
          arrr = np.full((length, 2), pad)
          cv2.fillPoly(image, pts=[np.subtract(arr, arrr)], color=(125, 125, 125))
      # Draw obstacle
      cv2.fillPoly(image, pts=[arr], color=(0, 0, 0))

  return image


# Set obstacles and circular padding
def setObstaclesAndCircularPadding(image, bounding_location, padding):
  # Set padding if padding is greater than 0
  if padding > 0:
    doPadding = True
    # Define a padding array
    paddings = [
        [0, padding],
        [0, -padding],
        [padding, 0],
        [-padding, 0],
                ]

  for obstacle in bounding_location:
    if len(obstacle) == 2:
      # Draw paddings
      if doPadding:
        for pad in paddings:
          cv2.rectangle(image, (obstacle[0][0] - pad[0], obstacle[0][1] - pad[1]), (obstacle[1][0] + pad[0], obstacle[1][1] + pad[1]), (125, 125, 125), -1)
        bound_points = sum(obstacle, [])
        points = [(bound_points[0], bound_points[1]), (bound_points[2], bound_points[3]), (bound_points[0], bound_points[3]), (bound_points[2], bound_points[1])]
        for point in points:
          cv2.circle(image, point, padding, (125, 125, 125), -1)
      # Draw obstacle
      cv2.rectangle(image, obstacle[0], obstacle[1], (0, 0, 0), -1)
    else:
      arr = np.array(obstacle, dtype=np.int32)
      # Draw paddings
      if doPadding:
        for pad in paddings:
          length = len(obstacle)
          arrr = np.full((length, 2), pad)
          cv2.fillPoly(image, pts=[np.subtract(arr, arrr)], color=(125, 125, 125))
        for point in obstacle:
          cv2.circle(image, tuple(np.int32(np.array(point))), padding, (125, 125, 125), -1)
      # Draw obstacle
      cv2.fillPoly(image, pts=[arr], color=(0, 0, 0))

  return image


# Define variables for the algorithm

# y size
height = 500 
# x size
width = 1200 
# Set padding length
padding = 0

timestep = 0

# Check if should create a video
recording = False

# Input variables
ptype = input("\nEnter padding type: ")
padding = int(input("\nEnter padding distance: "))
recording = input("\nEnter if video should be recorded, y/n")
if recording.lower() == 'y':
  print("Video will be recorded")
  recording = True
else:
  recording = False

obstacle_file_path = ""

# Define bounding boxes
obstacle_bounding_boxes = [
    [[175, 100], [100, 500] ], 
    [[275, 400], [350, 0]],
    [[650 - (75*(3**0.5)), 325], [650 - (75*(3**0.5)), 175], [650, 100], [650 + (75*(3**0.5)), 175], [650 + (75*(3**0.5)), 325], [650, 400]],
    [[900, 450], [1100, 450], [1100, 50], [900, 50], [900, 125], [1020, 125], [1020, 375], [900, 375]],
                            ]

# Create open list
open = PriorityQueue()

# Create a video writer
if recording:
  size = (width, height)
  fps = 90
  record = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)


# Create grid and backtrack array
if os.path.exists(obstacle_file_path) and os.path.isfile(obstacle_file_path):
  pass
else:
  # Enter array manually maybe through prompt
  grid = createGrid(height, width, obstacle_bounding_boxes, padding, ptype, True, padding)
  backtrack_grid = np.full((height*width), -1)


# Ensure valid starting points
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


# Ensure valid goal points
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


start = time.time()
# Main algorithm logic
while not open.empty():
  # Query the node with least cost
  explore = open.get()
  # Get current node location
  current_pos = explore[1]

  # Check if node has been visited
  if not grid[current_pos] == -13:
    timestep += 1
    grid[current_pos] = -13

    # Check if node is the goal node
    if current_pos == goal_index:
      last_explored = explore
      print("\nGoal path found")

      # Save frame as video
      if recording:
        # Image show
        data = np.copy(grid)
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
        record.write(image)

      break

    # x position for the current node
    x_pos = int(current_pos % width)
    # y position for the current node
    y_pos = int((current_pos - (current_pos % width))/width)
    # Define array to hold neighbours of the node
    neighbours_s = []
    neighbours_d = []

    # Set neighbours
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

    
    for neighbour in neighbours_s:
      # Add to open and grid only if not obstacle or padded area
      if grid[neighbour] < -10:
        continue
      cost = explore[0] + 1
      # Add neighbour to open list or update node
      if grid[neighbour] == -1 or grid[neighbour] > cost:
        grid[neighbour] = cost
        backtrack_grid[neighbour] = current_pos
        open.put((cost, neighbour))

    for neighbour in neighbours_d:
      # Add to open and grid only if not obstacle or padded area
      if grid[neighbour] < -10:
        continue
      cost = explore[0] + 2**0.5
      # Add neighbour to open list or update node
      if grid[neighbour] == -1 or grid[neighbour] > cost:
        grid[neighbour] = cost
        backtrack_grid[neighbour] = current_pos
        open.put((cost, neighbour))

    # Save frame as video
    if recording and (timestep < fps or timestep % 500 == 0):
      # Image show
      data = np.copy(grid)
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

      record.write(image)

# Print algorithm end time
timef = time.time() - start
print(f"Goal found in {math.floor(timef/60)} minutes and {(timef % 60):.2f} seconds")

# Check backtracking time
start = time.time()
index = goal_index

# Backtrack logic
while backtrack_grid[index] > 0:
  grid[index] = -4
  index = backtrack_grid[index]


# Save frame as video
if recording:
  # Image show
  data = np.copy(grid)
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

  for i in range(fps):
    record.write(image)

# Print backtrack end time
timef = time.time() - start
print(f"Backtracking done in {math.floor(timef/60)} minutes and {(timef % 60):.2f} seconds")


if recording:
  record.release()

# Display video of full algorithm
if recording:
  cap = cv2.VideoCapture("video.avi")

  if (cap.isOpened()== False):
    print("Error opening video stream or file")

  while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
      cv2.imshow("Djikstra", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    else:
      break

  cv2.destroyAllWindows()
  cap.release()