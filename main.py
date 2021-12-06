import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils
import math
from collections import defaultdict
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

path_to_image = 'photos/test5_1.jpg'
path_to_ship_detection_model = './models/ship_detection_model_2'


# Import image, resize, and convert to other useful formats
# img_orig = cv.imread('photos/testimg3.jpg')
img_orig = cv.imread(path_to_image)
img_high_res = cv.cvtColor(imutils.resize(img_orig, width=2000), cv.COLOR_BGR2RGB)
img_orig = imutils.resize(img_orig, width=1000)
img_gray = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)
img_h = img_hsv[:,:,0]

# Create image for drawing
annotated_img = img.copy()


############################################
# Define parameters for road and settlement searching
############################################
# Mask radius for roads and settlements
check_road_radius = 50
check_settlement_radius = 15

# Width and height of boxes used to detect roads within masked area
box_w = 70
box_h = 20

# Number of nonblack pixels required to consider looking for a settlement or road in an image
nonblack_px_thresh = 250

############################################
# Import models
############################################
ship_model = tf.keras.models.load_model(path_to_ship_detection_model)
ship_model.summary()

# save the model to a json format
model_json = ship_model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to h5
ship_model.save_weights('model.h5')

# export the whole model to h5
ship_model.save('full_model.h5')

ship_class_names = ['brick', 'question', 'sheep', 'stone', 'wheat', 'wood']

############################################
# Port detection
############################################

def detect_ports(img_in, annotate=False):
    # Get canny edges
    edges = cv.Canny(img,500,700)

    # Get blue mask

    # Create boundaries
    hsv = cv.cvtColor(img_in, cv.COLOR_RGB2HSV)
    blue_lower=np.array([80,140,200],np.uint8)
    blue_upper=np.array([140,255,255],np.uint8)

    # Find mask
    blue_mask = cv.inRange(hsv, blue_lower, blue_upper)

    # Dilate
    blue_mask = cv.dilate(blue_mask,(17,17),iterations=15)

    # Mask image with blue mask
    res = cv.bitwise_and(img_in,img_in, mask=blue_mask)

    ## Expand blue mask outwards

    # Create flood masks
    h, w = img_in.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)

    # Flood fill each corner to extract the board from the background
    cv.floodFill(blue_mask, flood_mask, (0,0), 255)
    cv.floodFill(blue_mask, flood_mask, (w-1,0), 255)
    cv.floodFill(blue_mask, flood_mask, (0,h-1), 255)
    cv.floodFill(blue_mask, flood_mask, (w-1,h-1), 255)

    # Get image of only board from mask
    blue_mask_inv = cv.bitwise_not(blue_mask)
    cropped_img = cv.bitwise_and(img_in,img_in, mask=blue_mask_inv)

    ## Get ship contours

    # Get center of mass of all pixels in blue mask
    mass_x, mass_y = np.where(blue_mask <= 0)
    cent_x = int(np.average(mass_x))
    cent_y = int(np.average(mass_y))

    # Get mask of all ships
    ship_flood_mask = np.zeros((h+2, w+2), np.uint8)
    ship_mask = blue_mask.copy()
    cv.floodFill(ship_mask, ship_flood_mask, (cent_x,cent_y), 255)
    ship_mask_inv = cv.bitwise_not(ship_mask)

    # Find contours in mask
    contours, hierarchy = cv.findContours(image=ship_mask_inv, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    # Sort contours in decreasing order and take largest 9, which should correspond to the ports
    contours = sorted(contours, key=lambda x:cv.contourArea(x), reverse=True)
    contours = contours[0:9]

    # Draw contours on annotated image
    if annotate:
        cv.drawContours(image=annotated_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    ship_img = cv.bitwise_and(img_in,img_in, mask=ship_mask_inv)

    ship_img_mask = np.where(ship_img==0,0,255)
    land_img = cropped_img.copy()
    land_img = np.where(ship_img!=0,0,land_img)

    return contours, land_img

_, land_img = detect_ports(img, True)
contours, _ = detect_ports(img_high_res)
# # see ship img
# plt.figure(figsize=(10, 10))
# plt.subplot(121),plt.imshow(annotated_img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(ship_img)
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()


# plt.figure(figsize=(6,6))
# plt.imshow(land_img)
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

## Extract port locations and feed each to model for detection

ports_extracted = []
ports_locations = []
# Get the bounding box of all the contours
port_detect_img = img.copy()
# port_detect_img = cv.copyMakeBorder(port_detect_img, 128, 128, 128, 128, cv.BORDER_REPLICATE)
for contour in contours:
    x,y,w,h = cv.boundingRect(contour)

    low_res_pt = (int(x/2), int(y/2))
    low_res_wh = (int(w/2), int(h/2))
    # cv.rectangle(img=annotated_img, pt1=low_res_pt, pt2=(low_res_pt[0]+low_res_wh[0],low_res_pt[1]+low_res_wh[1]), color=(255, 255, 0), thickness=2)

    # Get a rectange of 128x128 with the original rectangle at the center
    hor_diff = int((w-128)/2)
    ver_diff = int((h-128)/2)
    # print(f"hor diff, ver diff: {hor_diff}, {ver_diff}")
    print(f"w, h: {w}, {h}")
    print(f"ver_diff, h-ver_diff: {ver_diff}, {h-ver_diff}")
    print(f"hor_diff, w-hor_diff: {hor_diff}, {w-hor_diff}")
    print(f'ydiff, xdiff: {(h-ver_diff)-ver_diff}, {(w-hor_diff)-hor_diff}')
    ydiff = (h-ver_diff)-ver_diff
    xdiff = (w-hor_diff)-hor_diff

    # Draw the rectangle on the original image
    # cv.rectangle(img=annotated_img, pt1=(x+hor_diff,y+ver_diff), pt2=(x+hor_diff+128,y+ver_diff+128), color=(0, 255, 0), thickness=2)
    # cv.rectangle(img=annotated_img, pt1=(x+w-64,y+h-64), pt2=(x+w+64,y+h+64), color=(0, 255, 0), thickness=2)
    # cv.rectangle(img=annotated_img, pt1=(x+hor_diff,y+ver_diff), pt2=(x+w-hor_diff,y+h-ver_diff), color=(0, 255, 0), thickness=2)

    # Get the rectangle
    offsety = 128 - ydiff
    offsetx = 128 - xdiff
    ports_extracted.append(img_high_res[y+ver_diff:y+h-ver_diff+offsety, x+hor_diff:x+w-hor_diff+offsetx])
    ports_locations.append([int((x+w/2)/2),int((y+h/2)/2)])
# # show the image
# plt.figure(figsize=(6,6))
# plt.imshow(annotated_img)
# plt.show()

# # Show the extracted ports in a 9x9 grid
# for i in range(len(ports_extracted)):
#     plt.subplot(3,3,i+1),plt.imshow(ports_extracted[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# make model predictions on extracted ports
# show the extracted ports in a 9x9 grid
for i in range(len(ports_extracted)):

    # predict on the model
    prediction = ship_model.predict(np.expand_dims(ports_extracted[i], axis=0))

    idx = np.argmax(prediction)

    cls = ship_class_names[idx]

    # ax = plt.subplot(3,3,i+1)
    # ax.title.set_text(cls)

    loc = ports_locations[i]
    cv.putText(annotated_img, cls, (loc[0]-60, loc[1]-30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)

    # plt.imshow(ports_extracted[i])
    # plt.xticks([]), plt.yticks([])

plt.figure(figsize=(6,6))
plt.imshow(annotated_img)
plt.show()


############################################
# Find roads and settlements in images
############################################

# RGB ranges for the four player colors
blue_lower = np.array([80, 50, 80])
blue_upper = np.array([140, 255, 200])

orange_lower_hsv = np.array([15, 140, 150])
orange_upper_hsv = np.array([20, 255, 255])

white_lower = np.array([210, 210, 210])
white_upper = np.array([255, 255, 255])

red_lower = np.array([0, 140, 0])
red_upper = np.array([10, 255, 255])

# Compute masks for each color from original image
blue_mask = cv.inRange(img_hsv, blue_lower, blue_upper)
orange_mask = cv.inRange(img_hsv, orange_lower_hsv, orange_upper_hsv)
white_mask = cv.inRange(img_orig, white_lower, white_upper)
red_mask = cv.inRange(img_hsv, red_lower, red_upper)


# Use board image of only land from ship detection code to remove background
land_range = cv.inRange(land_img, np.array([0,0,0]),np.array([0,0,0]))
red_mask = np.where(land_range!=0,0,red_mask)
land_range = cv.inRange(land_img, np.array([0,0,0]),np.array([0,0,0]))
white_mask = np.where(land_range!=0,0,white_mask)

# Mask image to get residual pixels for corresponding road color
blue_res = cv.bitwise_and(img,img, mask=blue_mask)
orange_res = cv.bitwise_and(img,img, mask=orange_mask)
red_res = cv.bitwise_and(img,img, mask=red_mask)
white_res = cv.bitwise_and(img,img, mask=white_mask)

# Erode and dilate image to remove unwanted background noise in commonly offending images
white_mask = cv.morphologyEx(white_mask, cv.MORPH_OPEN, (7,7), iterations=5)
blue_res = cv.morphologyEx(blue_res, cv.MORPH_OPEN, (7,7), iterations=1)


# plt.figure(figsize=(30, 30))
# plt.subplot(121),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blue_res)
# plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# plt.figure(figsize=(30, 30))
# plt.subplot(121),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(orange_res)
# plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# plt.figure(figsize=(30, 30))
# plt.subplot(121),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(white_res)
# plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# plt.figure(figsize=(30, 30))
# plt.subplot(121),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(red_res)
# plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
# plt.show()




############################################
# Find circle for each number, bandit, and dice
############################################

# Find circles using hough transform
circles = cv.HoughCircles(img_gray,cv.HOUGH_GRADIENT_ALT,1,20,
                            param1=50,param2=.92,minRadius=10,maxRadius=50)
circles = np.uint16(np.around(circles))[0,:,0:3]
circles_img = img.copy()
for circ in circles:
    cv.circle(circles_img,(circ[0],circ[1]),circ[2],(255,255,255),-1)
# plt.figure(figsize=(11,11))
# plt.imshow(circles_img)
# plt.xticks([])
# plt.yticks([])
# Filter circles

# Compute average HSV of a set of points around and within each detected circle
filter_img = np.zeros(img.shape)
e = 10
num_pts = 10
avgs = []
avgs_inside = []
new_img_hsv = img_hsv.copy()
for circ in circles:
    cv.circle(filter_img, (circ[0], circ[1]), circ[2], (255, 255, 255), -1)
    cv.circle(filter_img, (circ[0], circ[1]), circ[2]+e, (0, 255, 0), 2)
    if circ[2] > 5:
        cv.circle(filter_img, (circ[0], circ[1]), circ[2]-5, (0, 255, 0), 2)
    avg = [0, 0, 0]
    avg_inside = [0, 0, 0]
    for j in range(1,num_pts+1):
        x = int(circ[0] + (e+circ[2])*np.cos(j*2*np.pi/10))
        y = int(circ[1] + (e+circ[2])*np.sin(j*2*np.pi/10))
        avg += img_hsv[y, x, :]
        cv.circle(filter_img, (x,y), 1, (255, 0, 0), 2)
        x1 = int(circ[0] + (-5+circ[2])*np.cos(j*2*np.pi/10))
        y1 = int(circ[1] + (-5+circ[2])*np.sin(j*2*np.pi/10))
        avg_inside += img_hsv[y1, x1, :]
        cv.circle(filter_img, (x1,y1), 1, (255, 0, 0), 2)
    avg = avg/num_pts
    avg_inside = avg_inside/num_pts
    avgs.append(avg)
    avgs_inside.append(avg_inside)
    # Display average HSV values on each tile center for use in determining a range of HSVs to use in detection
    # print(f"avg: {avg}")
    # cv.circle(new_img_hsv, (circ[0], circ[1]), circ[2], (int(avg[0]), int(avg[1]), int(avg[2])), -1)
    cv.circle(new_img_hsv, (circ[0], circ[1]), circ[2], (int(avg_inside[0]), int(avg_inside[1]), int(avg_inside[2])), -1)
    # cv.putText(new_img_hsv, "H: " + str(avg_inside[0]), (circ[0]-50, circ[1]), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    # cv.putText(new_img_hsv, "S: " + str(avg_inside[1]), (circ[0]-50, circ[1]+25), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    # cv.putText(new_img_hsv, "V: " + str(avg_inside[2]), (circ[0]-50, circ[1]+50), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    cv.putText(new_img_hsv, "H: " + str(avg[0]), (circ[0]-50, circ[1]), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    cv.putText(new_img_hsv, "S: " + str(avg[1]), (circ[0]-50, circ[1]+25), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    cv.putText(new_img_hsv, "V: " + str(avg[2]), (circ[0]-50, circ[1]+50), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
# print(f"averages: {avgs}")
# plt.figure(figsize=(11,11))
# plt.imshow(filter_img, cmap='gray')
# plt.title('filter image'), plt.xticks([]), plt.yticks([])
# plt.figure(figsize=(11,11))
# plt.imshow(new_img_hsv)
# plt.title('hsv image'), plt.xticks([]), plt.yticks([])

# Separate circles into numbers, dice, and bandit
bandit_loc = []
dice_locs = []
num_locs = []
for i, circ in enumerate(circles):
    hsv = avgs_inside[i]
    if hsv[0] >= 15 and hsv[0] <= 25 and \
        hsv[1] >= 40 and hsv[1] <= 50 and \
        hsv[2] >= 115 and hsv[2] <= 130:
        bandit_loc.append(circ[0:3])
    elif hsv[0] >= 12 and hsv[0] <= 30 and \
        hsv[1] >= 50 and hsv[1] <= 130 and \
        hsv[2] >= 160 and hsv[2] <= 250:
        num_locs.append(circ[0:3])
    elif hsv[0] >= 15 and hsv[0] <= 80 and \
        hsv[1] >= 180 and hsv[1] <= 210 and \
        hsv[2] >= 130 and hsv[2] <= 260:
        dice_locs.append(circ[0:3])

# print(f"len bandit_loc: {len(bandit_loc)}")
# print(f"len dice_locs: {len(dice_locs)}")
# print(f"len num_locs: {len(num_locs)}")

new_img = img.copy()

# Detect tiles based on HSV ranges
for i, circ in enumerate(num_locs):
    hsv = avgs[i]
    type = 'none'
    if hsv[0] >= 15 and hsv[0] <= 25 and \
        hsv[1] >= 100 and hsv[1] <= 165 and \
        hsv[2] >= 160 and hsv[2] <= 230:
        type = 'wheat'
    elif hsv[0] >= 0 and hsv[0] <= 27 and \
        hsv[1] >= 80 and hsv[1] <= 160 and \
        hsv[2] >= 100 and hsv[2] <= 160:
        type = 'brick'
    elif hsv[0] >= 60 and hsv[0] <= 145 and \
        hsv[1] >= 28 and hsv[1] <= 55 and \
        hsv[2] >= 100 and hsv[2] <= 160:
        type = 'stone'
    elif hsv[0] >= 26 and hsv[0] <= 75 and \
        hsv[1] >= 25 and hsv[1] <= 110 and \
        hsv[2] >= 75 and hsv[2] <= 130:
        type = 'wood'
    elif hsv[0] >= 32 and hsv[0] <= 45 and \
        hsv[1] >= 80 and hsv[1] <= 150 and \
        hsv[2] >= 120 and hsv[2] <= 200:
        type = 'sheep'
    cv.putText(annotated_img, type, (circ[0]-35, circ[1]+50), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

print(f"bandit loc: {bandit_loc}")
if len(bandit_loc) == 1:
    cv.putText(annotated_img, 'bandit', (bandit_loc[0][0]-35, bandit_loc[0][1]+50), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

for die_loc in dice_locs:
    cv.putText(annotated_img, 'die', (die_loc[0]-20, die_loc[1]+50), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

# plt.figure(figsize=(11,11))
# plt.imshow(new_img)
# plt.title('hsv image'), plt.xticks([]), plt.yticks([])
# plt.show()


#########################
# Create board graph
#########################

def distanceBetweenCircles(c1, c2):
    return math.sqrt(((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2))

def populate_map(map_to_pop, circs, img_c=None):
    for i in range(len(circs)):
        for j in range(len(circs)):
            if (distanceBetweenCircles(circs[i], circs[j]) >= average_distance - distance_range and distanceBetweenCircles(circs[i], circs[j]) <= average_distance + distance_range):
                map_to_pop[i].append(j)
                if img_c is not None:
                    cv.line(img_c, (int(circs[i][0]), int(circs[i][1])), (int(circs[j][0]), int(circs[j][1])), (255, 0, 0), 5)
    return map_to_pop, img_c

def angleBetweenCircles(c1, c2):
    delta_x = c2[0] - c1[0]
    delta_y = c2[1] - c1[1]
    return math.atan2(delta_y, delta_x)

## Compute average distance between tile centers
# Total number of distances between tile centers
num_short_distances = 3 * 6 + 4 * 6 + 6 * 5
all_distances = []

circles = np.array(circles, dtype=np.float64)

# Compute distances between each pair of tile centers
for i in range(len(circles)):
    for j in range(len(circles)):
        if (i != j):
            all_distances.append(distanceBetweenCircles(circles[i], circles[j]))

# Sort by shortest to longest distance
all_distances.sort()

# Sum distances between closest tiles
short_distances_sum = 0
for i in range(num_short_distances):
    short_distances_sum += all_distances[i]

# Compute average tile center distance
average_distance = short_distances_sum / num_short_distances
# Allowed distance error between centers
distance_range = 60

img_copy = img.copy()
circles_map = defaultdict(list)

# Populate circles_map with as many keys as there are tiles, where
# each value is a list of neighboring tiles
circles_map, img_copy = populate_map(circles_map, circles, img_copy)

# Find outer circles
outer_circles = []
inner_circles = []
outer_real_circles = []
current_outer_circle = None
top_most_outer_y = 9999
top_most_outer_real_y = 9999
top_most_inner_y = 9999
for key in circles_map:
    val = circles_map[key]
    # If current tile has three neighbors, it is a corner tile and we must find
    # the other possible directions in which roads and settlements can exist
    if (len(val) == 3):
        cur_angles = []
        for other_c in val:
            cur_angles.append((math.pi * 2) + angleBetweenCircles(circles[key], circles[other_c]))
        cur_angles.sort()
        if (cur_angles[1] - cur_angles[0] - (math.pi / 6) > 1):
            cur_angles[0] += (math.pi * 2)
        if (cur_angles[2] - cur_angles[1] - (math.pi / 6) > 1):
            cur_angles[2] -= (math.pi * 2)
        cur_angles.sort()
        
        # Add other angles
        maxAngle = cur_angles[2]
        for i in range(3):
            maxAngle += math.pi / 3
            newX = circles[key][0] + (average_distance * math.cos(maxAngle))
            newY = circles[key][1] + (average_distance * math.sin(maxAngle))
            cv.line(img_copy, (int(circles[key][0]), int(circles[key][1])), (int(newX), int(newY)), (0, 255, 0), 5)
            outer_circles.append([newX, newY])
            top_most_outer_y = min(top_most_outer_y, newY)
            if (i == 1):
                current_outer_circle = [newX, newY]
            if (i == 0):
                next_outer_circle = [newX, newY]

    # If current tile has five neighbors, it is an internal tile and we know all places
    # roads and settlements can exist
    elif (len(val) == 5):
        inner_circles.append(circles[key])
        top_most_inner_y = min(top_most_inner_y, circles[key][1])
    
    if (len(val) != 5):
        outer_real_circles.append(circles[key])
        top_most_outer_real_y = min(top_most_outer_real_y, circles[key][1])

# Get center circle
average_inner_x = 0
average_inner_y = 0
for c in inner_circles:
    average_inner_x += c[0]
    average_inner_y += c[1]
average_inner_x /= 6
average_inner_y /= 6

center_circle = [average_inner_x, average_inner_y]

# cv.circle(img_copy, (int(average_inner_x), int(average_inner_y)), 20, (0, 0, 255), -1)

## Numbering scheme starts at top of image and rotates clockwise, moving down
# one layer at every rotation. Ports, roads, and settlements are then mapped
# to their respective numbers. There are three circles in total - one around the water,
# one comprised of the outer most ring of tiles, and one comprised of the inner
# most ring of tiles

def sort_circles(c):
  return angleBetweenCircles(c, center_circle)

# Find top most circle in the water ring
outer_circles.sort(key=sort_circles)
top_most_outer_y_index = 0
num_outer_circles = len(outer_circles)
for i in range(len(outer_circles)):
    cur_c = outer_circles[i]
    if (cur_c[1] == top_most_outer_y):
        top_most_outer_y_index = i
        break
outer_circles += outer_circles

# Find top most circle on the outer land ring
outer_real_circles.sort(key=sort_circles)
top_most_outer_real_y_index = 0
num_outer_real_circles = len(outer_real_circles)
for i in range(len(outer_real_circles)):
    cur_c = outer_real_circles[i]
    if (cur_c[1] == top_most_outer_real_y):
        top_most_outer_real_y_index = i
        break
outer_real_circles += outer_real_circles

# Find top most circle on the inner land ring
inner_circles.sort(key=sort_circles)
top_most_inner_y_index = 0
num_inner_circles = len(inner_circles)
for i in range(len(inner_circles)):
    cur_c = inner_circles[i]
    if (cur_c[1] == top_most_inner_y):
        top_most_inner_y_index = i
        break
inner_circles += inner_circles

# Populate list of points for the location of each graph node in clockwise, inward spiraling order
outer_circles_num = []
points = []
point_num = 0
# Water ring
for i in range(top_most_outer_y_index, top_most_outer_y_index + num_outer_circles):
    cur_c = outer_circles[i]
    point_num += 1
    points.append([cur_c[0], cur_c[1]])
    outer_circles_num.append(i)

# Outer tile ring
for i in range(top_most_outer_real_y_index, top_most_outer_real_y_index + num_outer_real_circles):
    cur_c = outer_real_circles[i]
    point_num += 1
    points.append([cur_c[0], cur_c[1]])

# Inner tile ring
for i in range(top_most_inner_y_index, top_most_inner_y_index + num_inner_circles):
    cur_c = inner_circles[i]
    point_num += 1
    points.append([cur_c[0], cur_c[1]])

points.append([center_circle[0], center_circle[1]])

for i in range(len(points)):
    cur_c = points[i]
    cv.putText(annotated_img, str(i), (int(cur_c[0]), int(cur_c[1])), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

# Travel around graph and find all neighbors
neighbor_sets = {}
neighbors = []
for i in range(len(points)):
    neighbors = []
    color = list(np.random.random(size=3) * 256)
    for j in range(len(points)):
        if (distanceBetweenCircles(points[i], points[j]) >= average_distance - distance_range and distanceBetweenCircles(points[i], points[j]) <= average_distance + distance_range):
            neighbors.append(j)
            # cv.line(annotated_img, (int(points[i][0]), int(points[i][1])), (int(points[j][0]), int(points[j][1])), color, 5)
    neighbor_sets[i] = neighbors


## Create masks for each road and settlement location

check_loc_img = img.copy()
road_mask_all = np.zeros(img.shape)
settlement_mask_all = np.zeros(img.shape)

def get_point_from_angle_dist(orig, ang, dist):
    return (orig[0]+dist*math.cos(ang), orig[1]+dist*math.sin(ang))

road_settle_locs = []
# Find possible road locations
# road_locs = []
road_masks = []
neighbor_pairs = []
for num in neighbor_sets:
    first_pt = points[num]
    for neighbor in neighbor_sets[num]:
        road_mask = np.zeros(img.shape)
        second_pt = points[neighbor]

        # Compute center point between two tiles
        road_loc = (int((first_pt[0]+second_pt[0])/2), int((first_pt[1]+second_pt[1])/2))
        # road_locs.append({"pts":[first_pt, second_pt], "loc":road_loc, "mask":road_mask, "neighbors":[num, neighbor]})

        # Compute rectangle corner locations
        ang = angleBetweenCircles(first_pt, second_pt)
        dist = np.sqrt((box_h/2)**2 + (box_w/2)**2)
        alpha = math.atan2((box_h/2),(box_w/2))
        beta = np.pi/2-alpha
        ang1 = ang+beta
        ang2 = ang1+2*alpha
        ang3 = ang-beta
        ang4 = ang3-2*alpha
        pt1 = get_point_from_angle_dist(road_loc, ang1, dist)
        pt2 = get_point_from_angle_dist(road_loc, ang2, dist)
        pt3 = get_point_from_angle_dist(road_loc, ang3, dist)
        pt4 = get_point_from_angle_dist(road_loc, ang4, dist)
        box_pts = np.array([[pt1],[pt2],[pt4],[pt3]],np.int32)
        box_pts = box_pts.reshape((-1, 1, 2))

        # Draw rectangle mask on road_mask
        # cv.polylines(check_loc_img, [box_pts], True, (0,255,0), 2)
        cv.fillPoly(check_loc_img, [box_pts], (0,255,0))
        cv.fillPoly(road_mask, [box_pts], (255,255,255))
        cv.fillPoly(road_mask_all, [box_pts], (255,255,255))
        road_masks.append(road_mask)
        road_settle_locs.append({"pts":[first_pt, second_pt], "loc":road_loc, "mask":road_mask, "neighbors":[num, neighbor], "type":"road"})
        neighbor_pairs.append([first_pt, second_pt])

# Find possible settlement locations
# settlement_locs = []
settlement_masks = []
for num in neighbor_sets:
    first_pt = points[num]
    neighbors_copy = neighbor_sets[num]

    # Compute dists between all other neighbors of starting tile
    neighbor_pts = np.array(points)[np.array(neighbors_copy)]
    n_dists = []
    for i, n_1 in enumerate(neighbors_copy):
        for j, n_2 in enumerate(neighbors_copy):
            if (i < j):
                dist = distanceBetweenCircles(neighbor_pts[i], neighbor_pts[j])
                if dist < 200:
                    n_dists.append([n_1,n_2])
    # Mark centers of neighbor triplets made with starting tile and each pair of neighboring tiles
    for pair in n_dists:
        settlement_mask = np.zeros(img.shape)
        second_pt = points[pair[0]]
        third_pt = points[pair[1]]
        # Compute centroid between neighbor triplets
        settlement_loc = (int((first_pt[0]+second_pt[0]+third_pt[0])/3),int((first_pt[1]+second_pt[1]+third_pt[1])/3))
        cv.circle(check_loc_img, settlement_loc, 20, (0,0,255), -1)
        cv.circle(settlement_mask_all, settlement_loc, check_settlement_radius, (255,255,255), -1)
        cv.circle(settlement_mask, settlement_loc, check_settlement_radius, (255,255,255), -1)
        # settlement_locs.append({"pts":[first_pt, second_pt, third_pt], "loc":settlement_loc, "mask":settlement_mask, "neighbors":pair})
        road_settle_locs.append({"pts":[first_pt, second_pt, third_pt], "loc":settlement_loc, "mask":settlement_mask, "neighbors":pair, "type":"settlement"})
        settlement_masks.append(settlement_mask)

# plt.figure(figsize=(11,11))
# plt.imshow(check_loc_img)
# plt.xticks([])
# plt.yticks([])
# plt.figure(figsize=(11,11))
# plt.imshow(road_mask_all)
# plt.xticks([])
# plt.yticks([])
# plt.figure(figsize=(11,11))
# plt.imshow(settlement_mask_all)
# plt.xticks([])
# plt.yticks([])


# Create total residual image out of each individual color
all_res = white_res + orange_res + blue_res + red_res
all_res_road = np.where(road_mask_all==0,0,all_res)
all_res_settle = np.where(settlement_mask_all==0,0,all_res)
all_res_total = all_res_road + all_res_settle
# plt.figure(figsize=(11,11))
# plt.subplot(1,4,1), plt.imshow(all_res_road)
# plt.xticks([]), plt.yticks([])
# plt.subplot(1,4,2), plt.imshow(all_res_settle)
# plt.xticks([]), plt.yticks([])
# plt.subplot(1,4,3), plt.imshow(img)
# plt.xticks([]), plt.yticks([])
# plt.subplot(1,4,4), plt.imshow(all_res_total)
# plt.xticks([]), plt.yticks([])
# plt.show()

road_set_labeled_img = img.copy()

# Crue if c1 between c2_l and c2_h, false otherwise
def between_color_range(c1,c2_l,c2_h):
    if c1[0] > c2_l[0] and c1[1] > c2_l[1] and c1[2] > c2_l[2] and c1[0] < c2_h[0] and c1[1] < c2_h[1] and c1[2] < c2_h[2]:
        return True
    return False

for info in road_settle_locs:
    
    # Change mask based on if we're detecting roads or settlements
    road_pt = info["loc"]
    if info["type"] == "road":
        targeted_img = np.where(info["mask"]==0,0,all_res_total)[road_pt[1]-check_road_radius:road_pt[1]+check_road_radius,road_pt[0]-check_road_radius:road_pt[0]+check_road_radius]
    elif info["type"] == "settlement":
        targeted_img = np.where(info["mask"]==0,0,all_res_total)[road_pt[1]-check_settlement_radius:road_pt[1]+check_settlement_radius,road_pt[0]-check_settlement_radius:road_pt[0]+check_settlement_radius]

    # Count nonblack pixels
    total_pixels = targeted_img.shape[0] * targeted_img.shape[1]
    nonblack_px = np.count_nonzero(np.all(targeted_img!=[0,0,0],axis=2))

    # cv.circle(road_set_labeled_img, (info["loc"][0], info["loc"][1]), 10, (0,255,0), -1)

    label = "none"
    # Only search if something exists in img (there are enough nonblack pixels)
    if nonblack_px > nonblack_px_thresh:
        # Compute dominant color of nonblack pixels
        pixels = np.float32(targeted_img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        # Get second most dominant color since first is probably black
        dom = palette[np.argsort(counts)[-2]]
        # Make sure second most dominant color isn't black
        if dom[0] < 5 and dom[1] < 5 and dom[2] < 5:
            # If it is, take first most dominant color
            dom = palette[np.argsort(counts)[-1]]

        # Define RGB color thresholds for roads and settlements
        orange_rgb_l = [150,100,0]
        orange_rgb_h = [255,180,100]

        blue_rgb_l = [0,50,0]
        blue_rgb_h = [75,95,255]

        white_rgb_l = [200,200,200]
        white_rgb_h = [255,255,255]

        red_rgb_l = [100,20,15]
        red_rgb_h = [230,80,75]

        if between_color_range(dom, orange_rgb_l, orange_rgb_h):
            label = "O"
            if info["type"] == "road":
                label += "R"
            elif info["type"] == "settlement":
                label += "S"
        elif between_color_range(dom, blue_rgb_l, blue_rgb_h):
            label = "B"
            if info["type"] == "road":
                label += "R"
            elif info["type"] == "settlement":
                label += "S"
        elif between_color_range(dom, white_rgb_l, white_rgb_h):
            label = "W"
            if info["type"] == "road":
                label += "R"
            elif info["type"] == "settlement":
                label += "S"
        elif between_color_range(dom, red_rgb_l, red_rgb_h):
            label = "R"
            if info["type"] == "road":
                label += "R"
            elif info["type"] == "settlement":
                label += "S"

        # if info["type"] == "settlement":
        #     targeted_img = cv.resize(targeted_img, (200,200))
            # cv.putText(targeted_img, label, (10, 10), cv.FONT_HERSHEY_SIMPLEX, .3, (255,255,255), 1)
            # cv.putText(targeted_img, str(dom[0]), (10, 20), cv.FONT_HERSHEY_SIMPLEX, .3, (255,255,255), 1)
            # cv.putText(targeted_img, str(dom[1]), (10, 30), cv.FONT_HERSHEY_SIMPLEX, .3, (255,255,255), 1)
            # cv.putText(targeted_img, str(dom[2]), (10, 40), cv.FONT_HERSHEY_SIMPLEX, .3, (255,255,255), 1)
            # plt.figure(figsize=(7,7))
            # plt.imshow(targeted_img)
            # plt.xticks([])
            # plt.yticks([])

        cv.putText(annotated_img, label, (info["loc"][0]-40, info["loc"][1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

plt.figure(figsize=(11,11))
plt.imshow(annotated_img), plt.xticks([]), plt.yticks([])
plt.show()