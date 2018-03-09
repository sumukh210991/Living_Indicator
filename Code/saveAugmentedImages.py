###########################################################################################
# Code for random augmented samples of all images
#-----------------------------------------------------------------------------------------

import random

orig_files = ["/home/student/Sumukh/Living_Indicator/img/file" + str(i) + ".png" for i in range(0, 5654)]
record = np.empty(0)
count = 0
for img_counter in range(5000, 5655):
    img = cv2.imread(orig_files[img_counter], 0)

    xpts = np.array([random.randint(100, 540) for p in range(0, 30)])
    ypts = np.array([random.randint(100, 540) for p in range(0, 30)])

    coords = np.vstack((xpts, ypts))
    coords = np.transpose(coords)

    subsetcount = 0
    for coord in coords:
        for _ in range(0, 3):
            top = random.randint(50, 100)
            bottom = random.randint(50, 100)
            left = random.randint(50, 100)
            right = random.randint(50, 100)
            # temp = np.array([coord, np.array([top, right, bottom, left])])
            temp = np.array([coord[0], coord[1], top, right, bottom, left])
            if (count == 0):
                record = temp
                filename = "/home/student/Sumukh/augmented_images/raw/raw_aug_file_" + str(img_counter) + "_" + str(
                    subsetcount) + ".png"
                count = count + 1
                subsetcount = subsetcount + 1
            else:
                record = np.vstack((record, temp))
                filename = "/home/student/Sumukh/augmented_images/raw/raw_aug_file_" + str(img_counter) + "_" + str(
                    subsetcount) + ".png"
                count = count + 1
                subsetcount = subsetcount + 1

            cv2.imwrite(filename, img[(coord[0] - left): (coord[0] + right), (coord[1] - bottom): (coord[1] + top)])