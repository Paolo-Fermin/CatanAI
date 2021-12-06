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
path_to_num_detection_model = './models/white_edges_classifier_2'

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
# num_model = tf.keras.models.load_model(path_to_ship_detection_model)
# num_model.summary()

# # save the model to a json format
# model_json = num_model.to_json()

# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# # serialize weights to h5
# num_model.save_weights('model.h5')

# # export the whole model to h5
# num_model.save('full_model.h5')


# num_model = tf.keras.models.load_model(path_to_ship_detection_model)
# num_model.summary()

# # save the model to a json format
# model_json = num_model.to_json()

# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# # serialize weights to h5
# num_model.save_weights('model.h5')

# # export the whole model to h5
# num_model.save('full_model.h5')


num_class_names = ['eight', 'eleven', 'five', 'four', 'nine', 'six', 'ten', 'three', 'twelve', 'two']


num_model = tf.keras.models.load_model(path_to_num_detection_model)
# Check its architecture
num_model.summary()

# save the model to a json format
model_json = num_model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to h5
model.save_weights('model.h5')
# export the whole model to h5
model.save('full_model.h5')


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
        hsv[1] >= 80 and hsv[1] <= 160 and \
        hsv[2] >= 120 and hsv[2] <= 200:
        type = 'sheep'
    cv.putText(annotated_img, type, (circ[0]-35, circ[1]+50), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

if len(bandit_loc) == 1:
    cv.putText(annotated_img, 'bandit', (bandit_loc[0][0]-35, bandit_loc[0][1]+50), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

for die_loc in dice_locs:
    cv.putText(annotated_img, 'die', (die_loc[0]-20, die_loc[1]+50), cv.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

# print(num_locs)

nums_extracted = []
nums_locations = []
# Get the bounding box of all the contours
port_detect_img = img.copy()
# port_detect_img = cv.copyMakeBorder(port_detect_img, 128, 128, 128, 128, cv.BORDER_REPLICATE)
for circ in num_locs:
    cv.circle(annotated_img,(circ[0],circ[1]),circ[2],(255,255,255),-1)
    # cv.circle(img=annotated_img, pt1=low_res_pt, pt2=(low_res_pt[0]+low_res_wh[0],low_res_pt[1]+low_res_wh[1]), color=(255, 255, 0), thickness=2)
    highrescircle = img_high_res[circ[0]-64:circ[0]+64, circ[1]-64:circ[1]+64]
    nums_extracted.append(highrescircle)
    nums_locations.append([int(x/2),int(y/2)])
    plt.figure(figsize=(6,6))
    plt.imshow(highrescircle)
    plt.show()
    # cv.rectangle(img=annotated_img, pt1=(int(x/2)-64,y-64), pt2=(x+64,y+64), color=(255, 255, 0), thickness=2)
# # show the image

# # Show the extracted nums in a 3x6 grid
# for i in range(len(nums_extracted)):
#     plt.subplot(3,6,i+1),plt.imshow(nums_extracted[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# # make model predictions on extracted ports
# # show the extracted ports in a 9x9 grid
# port_results = []
# for i in range(len(nums_extracted)):

#     # predict on the model
#     prediction = num_model.predict(np.expand_dims(nums_extracted[i], axis=0))

#     idx = np.argmax(prediction)

#     cls = num_class_names[idx]

#     # ax = plt.subplot(3,3,i+1)
#     # ax.title.set_text(cls)

#     loc = nums_locations[i]
#     port_results.append(cls)
#     cv.putText(annotated_img, cls, (loc[0]-60, loc[1]-30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)

#     # plt.imshow(nums_extracted[i])
#     # plt.xticks([]), plt.yticks([])

# plt.figure(figsize=(6,6))
# plt.imshow(annotated_img)
# plt.show()
