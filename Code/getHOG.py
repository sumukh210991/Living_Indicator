def get_imgrad(filenames):
    features = []
    gradhist = []
    for i in range(0, len(filenames)):
        image = cv2.imread(filenames[i], 0)
        #chans = cv2.split(image)
        # gradhist = []

        laplace = cv2.Laplacian(image, cv2.CV_32F)
        #sobelx = cv2.Sobel(chan, cv2.CV_32F, 1, 0, ksize=5)
        #sobely = cv2.Sobel(chan, cv2.CV_32F, 0, 1, ksize=5)
        hist = cv2.calcHist([laplace], [0], None, [16], [0, 256]) #, sobely, laplace,
        hist = cv2.normalize(np.array(hist), dst=cv2.NORM_MINMAX)
        gradhist.extend(hist)
        #features.append(np.array(gradhist).flatten())
    feature = np.array(np.array(gradhist).flatten())
    feature = feature.reshape((len(filenames), 16))
    return feature