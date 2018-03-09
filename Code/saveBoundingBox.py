########################################################################################################
# Code For bounding box based CNN features - whole image
#-------------------------------------------------------------------------------------------------------

import numpy as np
import cv2


count = 0
feat = np.empty(0)

pts2 = np.float32([[0, 0], [0, 640], [640, 0], [640, 640]])
pts1 = np.float32([[144, 60], [144, 425], [513, 60], [513, 425]])

M = cv2.getPerspectiveTransform(pts1, pts2)


# for loop here
for i in range(0, 5654):
    house_mask = cv2.imread('/home/student/Sumukh/Results/House/test_house'+ str(i) + '.png')
    road_mask = cv2.imread('/home/student/Sumukh/Results/Road/test_road'+ str(i) + '.png')
    tree_mask = cv2.imread('/home/student/Sumukh/Results/Trees/test_tree'+ str(i) + '.png')
    terrain_mask = cv2.imread('/home/student/Sumukh/Results/Terrain/test_terrain'+ str(i) + '.png')

    image = cv2.imread('/home/student/Sumukh/Living_Indicator/img/file' + str(i)+ '.png')

    road_mask = cv2.warpPerspective(road_mask, M, (640, 640))
    house_mask = cv2.warpPerspective(house_mask, M, (640, 640))
    tree_mask = cv2.warpPerspective(tree_mask, M, (640, 640))
    terrain_mask = cv2.warpPerspective(terrain_mask, M, (640, 640))

    road_mask = cv2.cvtColor(road_mask, cv2.COLOR_BGR2GRAY)
    house_mask = cv2.cvtColor(house_mask, cv2.COLOR_BGR2GRAY)
    tree_mask = cv2.cvtColor(tree_mask, cv2.COLOR_BGR2GRAY)
    terrain_mask = cv2.cvtColor(terrain_mask, cv2.COLOR_BGR2GRAY)

    _, road_mask = cv2.threshold(road_mask, 30, 255, cv2.THRESH_BINARY)
    _, house_mask = cv2.threshold(house_mask, 30, 255, cv2.THRESH_BINARY)
    _, tree_mask = cv2.threshold(tree_mask, 30, 255, cv2.THRESH_BINARY)
    _, terrain_mask = cv2.threshold(terrain_mask, 30, 255, cv2.THRESH_BINARY)

    road_x,road_y,road_w,road_h = cv2.boundingRect(road_mask)
    house_x,house_y,house_w,house_h = cv2.boundingRect(house_mask)
    tree_x,tree_y,tree_w,tree_h = cv2.boundingRect(tree_mask)
    terrain_x,terrain_y,terrain_w,terrain_h = cv2.boundingRect(terrain_mask)

    road_pts = np.float32([[road_x, road_y], [road_x, road_y+road_h], [road_x+road_w, road_y], [road_x+road_w, road_y+road_h]])
    tree_pts = np.float32([[tree_x, tree_y], [tree_x, tree_y+tree_h], [tree_x+tree_w, tree_y], [tree_x+tree_w, tree_y+tree_h]])
    house_pts = np.float32([[house_x, house_y], [house_x, house_y+house_h], [house_x+house_w, house_y], [house_x+house_w, house_y+house_h]])
    terrain_pts = np.float32([[terrain_x, terrain_y], [terrain_x, terrain_y+terrain_h], [terrain_x+terrain_w, terrain_y], [terrain_x+terrain_w, terrain_y+terrain_h]])

    road_M = cv2.getPerspectiveTransform(road_pts, pts2)
    tree_M = cv2.getPerspectiveTransform(tree_pts, pts2)
    house_M = cv2.getPerspectiveTransform(house_pts, pts2)
    terrain_M = cv2.getPerspectiveTransform(terrain_pts, pts2)

    road = cv2.warpPerspective(image, road_M, (640, 640))
    tree = cv2.warpPerspective(image, tree_M, (640, 640))
    house = cv2.warpPerspective(image, house_M, (640, 640))
    terrain = cv2.warpPerspective(image, terrain_M, (640, 640))

    
'''# Code for saving the bounded box per each category

    road_filename = "/home/student/Sumukh/augmented_images/bounding_box/road_bounding_box/bd_box_road_" + str(
        i) + ".png"
    tree_filename = "/home/student/Sumukh/augmented_images/bounding_box/tree_bounding_box/bd_box_tree_" + str(
        i) + ".png"
    house_filename = "/home/student/Sumukh/augmented_images/bounding_box/house_bounding_box/bd_box_house_" + str(
        i) + ".png"
    terrain_filename = "/home/student/Sumukh/augmented_images/bounding_box/terrain_bounding_box/bd_box_terrain_" + str(
        i) + ".png"

    cv2.imwrite(road_filename, road)
    cv2.imwrite(tree_filename, tree)
    cv2.imwrite(house_filename, house)
    cv2.imwrite(terrain_filename, terrain)
    
    '''