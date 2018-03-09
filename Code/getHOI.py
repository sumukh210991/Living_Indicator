def get_colhist(filenames):
    features = []
    colhist = []
    for i in range(0, len(filenames)):
        image = cv2.imread(filenames[i], 0)
        image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist(image, [0], None, [16], [0, 256])
        hist = cv2.normalize(np.array(hist), dst=cv2.NORM_MINMAX)
        colhist.extend(hist)
        #features.append(np.array(colhist).flatten())
    features = np.array(np.array(colhist).flatten())
    features = features.reshape((len(filenames), 16))
    return features