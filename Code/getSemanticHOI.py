#############################################################################################################
# Segmented image analysis
#############################################################################################################

seg_files_house = ["/home/student/Sumukh/Results/House/test_house"+ (str(i)) + ".png" for i in range(0,5654)]
seg_files_road = ["/home/student/Sumukh/Results/Road/test_road"+ (str(i)) + ".png" for i in range(0,5654)]
seg_files_tree = ["/home/student/Sumukh/Results/Trees/test_tree"+ (str(i)) + ".png" for i in range(0,5654)]
seg_files_terrain = ["/home/student/Sumukh/Results/Terrain/test_terrain"+ (str(i)) + ".png" for i in range(0,5654)]
orig_files = ["/home/student/Sumukh/Living_Indicator/img/file" + str(i) + ".png" for i in range(0,5654)]


seg_feat_house = seg_feat_tree = seg_feat_terrain = seg_feat_road = np.empty(0)

for i in range(0, len(orig_files)):
    house = cv2.imread(seg_files_house[i])
    road = cv2.imread(seg_files_road[i])
    tree = cv2.imread(seg_files_tree[i])
    terrain = cv2.imread(seg_files_terrain[i])

    img1 = cv2.imread(orig_files[i])

    pts1 = np.float32([[144, 60], [144, 425], [513, 60], [513, 425]])
    pts2 = np.float32([[0, 0], [0, 640], [640, 0], [640, 640]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    house = cv2.warpPerspective(house, M, (640, 640))
    road = cv2.warpPerspective(road, M, (640, 640))
    tree = cv2.warpPerspective(tree, M, (640, 640))
    terrain = cv2.warpPerspective(terrain, M, (640, 640))

    house_seg = cv2.cvtColor(house, cv2.COLOR_BGR2GRAY)
    road_seg = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    tree_seg = cv2.cvtColor(tree, cv2.COLOR_BGR2GRAY)
    terrain_seg = cv2.cvtColor(terrain, cv2.COLOR_BGR2GRAY)

    house_ret, house_mask = cv2.threshold(house_seg, 30, 255, cv2.THRESH_BINARY)
    road_ret, road_mask = cv2.threshold(road_seg, 30, 255, cv2.THRESH_BINARY)
    tree_ret, tree_mask = cv2.threshold(tree_seg, 30, 255, cv2.THRESH_BINARY)
    terrain_ret, terrain_mask = cv2.threshold(terrain_seg, 30, 255, cv2.THRESH_BINARY)

    #house_img_bg = cv2.bitwise_and(img1, img1, mask=house_mask)
    #road_img_bg = cv2.bitwise_and(img1, img1, mask=road_mask)
    #tree_img_bg = cv2.bitwise_and(img1, img1, mask=tree_mask)
    #terrain_img_bg = cv2.bitwise_and(img1, img1, mask=terrain_mask)

    if(i == 0):
        seg_feat_house = get_segmented_color_hist(img1, house_mask)
        seg_feat_road = get_segmented_color_hist(img1, road_mask)
        seg_feat_tree = get_segmented_color_hist(img1, tree_mask)
        seg_feat_terrain = get_segmented_color_hist(img1, terrain_mask)
    else:
        seg_feat_house = np.vstack((seg_feat_house, get_segmented_color_hist(img1, house_mask)))
        seg_feat_road = np.vstack((seg_feat_road, get_segmented_color_hist(img1, road_mask)))
        seg_feat_tree = np.vstack((seg_feat_tree, get_segmented_color_hist(img1, tree_mask)))
        seg_feat_terrain = np.vstack((seg_feat_terrain, get_segmented_color_hist(img1, terrain_mask)))



# np.savetxt("/home/student/Sumukh/Living_Indicator/feat_house.csv",seg_feat_house, delimiter = ',', newline = '\n')
# seg_feat_house = np.loadtxt("/home/student/Sumukh/Living_Indicator/feat_house.csv", delimiter = ',')
