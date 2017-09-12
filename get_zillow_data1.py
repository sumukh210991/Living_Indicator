# import pyzillow
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults
import csv
import requests
import urllib
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from matplotlib import pyplot as plt
import cv2

def get_zillow_data(addr_list, zip_list):
    meta = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&"
    myloc = "C:\\Users\\sumukh210991\\Desktop\\Study\\Thesis\\popularity_score_for_houses\\python\\img"
    key = "&key=" + "AIzaSyC7-APuKb-aknoKymJdflh2jTC91HBe8rY"
    data = []
    count = 4960 # 4959
    for i in range(0, len(addr_list)):
        address = addr_list[i]
        zipcode = zip_list[i]
        zillow_data = ZillowWrapper("X1-ZWz1fpv5bng2rv_799rh")  # nishant_masc 'X1-ZWz19gfhe5z8y3_8xo7s'
        # nishant:'X1-ZWz1fk2vm7fkln_8tgid';
        # sumukh: 'X1-ZWz1fijmr1w9hn_9fxlx';
        # Sumukh@sdsu: 'X1-ZWz19anfkueyob_77v70';
        # other: 'X1-ZWz1fpv5bng2rv_799rh';
        try:
            deep_search_response = zillow_data.get_deep_search_results(address, zipcode)
            result = GetDeepSearchResults(deep_search_response)

            loc = str(result.latitude) + "," + str(result.longitude)
            print(loc)
            params = {"location": loc,
                      "width": "600",
                      "height": "400",
                      "key": "AIzaSyC7-APuKb-aknoKymJdflh2jTC91HBe8rY",
                      "fov": "90",
                      "pitch": "0",
                      "heading": "1"}
            metaurl = meta + urllib.urlencode(params)
            MyUrl = base + urllib.urlencode(params)
            file_loc = "img/file" + (str(count)) + ".png"
            print(file_loc)
            status = requests.get(metaurl)
            if ("ZERO_RESULTS" in str(status.content)):
                print("NO CONTENT")
                continue
            else:
                count = count + 1
                res = urllib.urlretrieve(MyUrl, os.path.join(file_loc))
                data.append(result)
        except:
            # data.append("none")
            continue
    return (data)


def GetStreet(Address, SaveLoc):
    meta = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&"
    params = {"location": Address,
              "width": "600",
              "height": "400",
              "key": "AIzaSyC7-APuKb-aknoKymJdflh2jTC91HBe8rY",
              "fov": "90",
              "pitch": "0",
              "heading": "1"}
    metaurl = meta + urllib.urlencode(params)
    MyUrl = base + urllib.urlencode(params)
    #fi = SaveLoc + r"\myfile.png"
    status = requests.get(metaurl)
    if ("ZERO_RESULTS" in str(status.content)):
        print("TRUE")
    else:
        print("FALSE")
    print(status.content)
    print(status)
    res = urllib.urlretrieve(MyUrl, os.path.join(fi))
    print(res)
    return res


def get_Living_Index(res):
    zestimate = res['zestimate_amount']
    property_size = res['property_size']
    home_size = res['home_size']

    zestimate = zestimate.fillna(np.median(zestimate[np.isnan(zestimate) == False]))
    property_size = property_size.fillna(np.median(property_size[np.isnan(property_size) == False]))
    home_size = home_size.fillna(np.median(home_size[np.isnan(home_size) == False]))

    zestimate = np.array(zestimate)
    property_size = np.array(property_size)
    home_size = np.array(home_size)

    builtin_ratio = property_size / home_size


    # Normalizing zestimate and builtin_ratio for PCA
    zestimate1 = np.array([0 + ((i - min(zestimate)) * 1) / (max(zestimate) - min(zestimate)) for i in zestimate])
    builtin_ratio1 = np.array(
        [0 + ((i - min(builtin_ratio)) * 1) / (max(builtin_ratio) - min(builtin_ratio)) for i in builtin_ratio])

    original_features = np.vstack((zestimate1, builtin_ratio1)).T

    arr_orig_features = np.array(original_features)

    score_pca = PCA(n_components=1)
    score = score_pca.fit_transform(arr_orig_features)

    for i in range(0, len(score)):
        if (score[i] > 0.054): # threshold for zillow_id = 1000000.0, property_size[i] / home_size[i] = 18
            score[i] = 0.054

    finalscore = np.array([0 + ((i - min(score)) * 10) / (max(score) - min(score)) for i in score])

    return finalscore

def get_colhist(filenames):
    features = []
    colhist = []
    for i in range(0, len(filenames)):
        image = cv2.imread(filenames[i])
        chans = cv2.split(image)
        #colhist = []
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [16], [0, 256])
            hist = cv2.normalize(np.array(hist), dst=cv2.NORM_MINMAX)
            colhist.extend(hist)
        #features.append(np.array(colhist).flatten())
    features = np.array(np.array(colhist).flatten())
    features = features.reshape((len(filenames), 48))
    return features


def get_imgrad(filenames):
    features = []
    gradhist = []
    for i in range(0, len(filenames)):
        image = cv2.imread(filenames[i])
        chans = cv2.split(image)
        # gradhist = []
        for chan in chans:
            laplace = cv2.Laplacian(chan, cv2.CV_32F)
            #sobelx = cv2.Sobel(chan, cv2.CV_32F, 1, 0, ksize=5)
            #sobely = cv2.Sobel(chan, cv2.CV_32F, 0, 1, ksize=5)
            hist = cv2.calcHist([laplace], [0], None, [32], [0, 256]) #, sobely, laplace,
            hist = cv2.normalize(np.array(hist), dst=cv2.NORM_MINMAX)
            gradhist.extend(hist)
        #features.append(np.array(gradhist).flatten())
    feature = np.array(np.array(gradhist).flatten())
    feature = feature.reshape((len(filenames), 96))
    return feature



'''def get_segmented_color_hist(img):
    hist_color_car = np.empty(0)
    count = 0
    for k in range(0,3):
        hist_color_car_chan = np.zeros(16)
        for i in range (0,img.shape[0]):
            for j in range(0, img.shape[1]):
                if(img[i,j,k] == 0):
                    count += 1
                    continue
                else:
                    hist_color_car_chan[int(img[i,j,k] // 16) - 1] += 1
        hist_color_car = np.append(hist_color_car , hist_color_car_chan / sum(hist_color_car_chan))
    return hist_color_car '''

def get_segmented_color_hist(img, mask):
    colhist = np.empty(0)
    #image = cv2.imread(img)
    chans = cv2.split(img)
    for chan in chans:
        hist = cv2.calcHist([chan], [0], mask, [16], [0, 256])
        hist = cv2.normalize(np.array(hist), dst=cv2.NORM_MINMAX)
        colhist = np.append(colhist, hist)
    return colhist



def gradient_descent(alpha, trainx, trainy, testx, testy, numiter, epsilon):
    theta = np.zeros(len(trainx[0]))
    m = len(trainy)
    arrcost = []
    for _ in range(1,numiter):
        pred = np.dot(trainx, theta)
        temp = np.dot((pred - trainy), trainx)
        theta = theta - ((alpha / m) * temp)
        val = np.sum((np.dot(trainx,theta) - trainy) / m)
        arrcost.append(np.abs(val))
        if(np.abs(val) < epsilon):
            break
            #return {'cost': arrcost, 'theta': theta }

    pred = np.dot(testx, theta)
    error = np.true_divide(len(pred[np.round(res) == testy]), len(testy)) * 100
    #return {'cost': arrcost, 'theta': theta}
    return error


def decision_tree(trainx, trainy, testx, testy):
    features = 'auto'
    depth = 5
    model = tree.DecisionTreeClassifier(max_features = features, max_depth= depth)
    model = model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy


def Support_vector_class(trainx, trainy, testx, testy):
    model = svm.SVC(decision_function_shape='ovo')
    model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy


def Linear_SVC(trainx, trainy, testx, testy):
    model = svm.LinearSVC()
    model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy


def neural_net(trainx, trainy, testx, testy):
    model = MLPClassifier(activation='tanh', solver ='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(trainx[0]),2), random_state=1)
    model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy


def naive_bayes(trainx, trainy, testx, testy):
    model = GaussianNB()
    model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy


# with open('LA.csv', 'rU') as infile:
with open('portland_metro.csv', 'rU') as infile:
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

# extract the variables you want
address = data['ADDRESS']
zip = data['ZIP']

address_all = []
zip_all = data['POSTCODE']
for i in range(0, len(data['NUMBER'])):
    address_all.append(data['NUMBER'][i] + ", " + data['STREET'][i] + " " + data['UNIT'][i])

# rand_idx = np.random.permutation(len(zip_all))

# random permutations saved in CSV for reuse in future : steps of 2000
np.savetxt("randperm.csv", rand_idx, delimiter= ",")


rand_idx = np.genfromtxt('randperm.csv',delimiter = ',')

address = []
zip = []
for i in rand_idx[12000:14000]: # last run until 14000
    address.append(address_all[int(i)])
    zip.append(zip_all[int(i)])

res = get_zillow_data(address, zip) # 721, 1487, 2282, 3323, 4165, 4960, 5656


# USE THIS FOR ZILLOW RESULTS
#
# import pickle
#
# f = open('result.pickle', 'wb')
# pickle.dump(res, f, 2)
# f.close()
#
# f = open('result.pickle1234', 'rb')
# res = pickle.load(f)
# f.close()

res = pd.read_csv('resultset_12000')
living_index = get_Living_Index(res)

filenames = []
for i in range(0,len(res)):
    name = "img/file"+str(i)+".png"
    filenames.append(name)



seg_feat_house = np.loadtxt("/home/student/Sumukh/Living_Indicator/feat_house.csv", delimiter = ',')
seg_feat_road = np.loadtxt("/home/student/Sumukh/Living_Indicator/feat_road.csv", delimiter = ',')
seg_feat_tree = np.loadtxt("/home/student/Sumukh/Living_Indicator/feat_tree.csv", delimiter = ',')
seg_feat_terrain = np.loadtxt("/home/student/Sumukh/Living_Indicator/feat_terrain.csv", delimiter = ',')

feature = get_imgrad(filenames)

#y = np.genfromtxt('pcascores.csv',delimiter = ',')
y = living_index
x = feature
# x = np.column_stack((np.ones(len(feature)), feature))

trainx = x[0:4523]
testx = x[4523:]
trainy = y[0:4523]
testy = y[4523:]

gd_accuracy = gradient_descent(0.001, trainx, trainy, testx, testy, 1000000, 0.001)

dTree_accuracy = decision_tree(trainx, trainy, testx, testy)

svc_accuracy = Support_vector_class(trainx, trainy, testx, testy)

lin_svc_accuracy = Linear_SVC(trainx, trainy, testx, testy)

nn_accuarcy = neural_net(trainx, trainy, testx, testy)

naive_bayes_accuarcy = naive_bayes(trainx, trainy, testx, testy)

# res = np.dot(testx, gd_res['theta'])
# np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100

# image = cv2.imread(r"img\file370.png")
# chans = cv2.split(image)
# colors = ("b", "g", "r")
# plt.figure()
# plt.title("'Flattened' Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# colhist = []
# features = []
# # loop over the image channels
# for (chan, color) in zip(chans, colors):
#     # create a histogram for the current channel and
#     #  concatenate the resulting histograms for each
#     # channel
#     hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
#     colhist.extend(hist)
#     # plot the histogram
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])
# features.extend(np.array(colhist).flatten())


import pandas as pd

dfresults = pd.DataFrame({'zillow_id':[],
                            'home_type':[],
                            'home_detail_link':[],
                            'graph_data_link':[],
                            'latitude':[],
                            'longitude':[],
                            'tax_year':[],
                            'tax_value':[],
                            'year_built':[],
                            'property_size':[],
                            'home_size':[],
                            'bathrooms':[],
                            'bedrooms':[],
                            'last_sold_date':[],
                            'last_sold_price':[],
                            'zestimate_amount':[],
                            'zestimate_last_updated':[],
                            'zestimate_value_change':[],
                            'zestimate_valuation_range_high':[],
                            'zestimate_valuationRange_low':[],
                            'zestimate_percentile':[]
                            })


j = -1
for res in temp_res:
    j+=1
    dfresultappend = pd.DataFrame({'zillow_id':res.zillow_id,
                                'home_type':res.home_type,
                                'home_detail_link':res.home_detail_link,
                                'graph_data_link':res.graph_data_link,
                                'latitude':res.latitude,
                                'longitude':res.longitude,
                                'tax_year':res.tax_year,
                                'tax_value':res.tax_value,
                                'year_built':res.year_built,
                                'property_size':res.property_size,
                                'home_size':res.home_size,
                                'bathrooms':res.bathrooms,
                                'bedrooms':res.bedrooms,
                                'last_sold_date':res.last_sold_date,
                                'last_sold_price':res.last_sold_price,
                                'zestimate_amount':res.zestimate_amount,
                                'zestimate_last_updated':res.zestimate_last_updated,
                                'zestimate_value_change':res.zestimate_value_change,
                                'zestimate_valuation_range_high':res.zestimate_valuation_range_high,
                                'zestimate_valuationRange_low':res.zestimate_valuationRange_low,
                                'zestimate_percentile':res.zestimate_percentile}, index = [j])
    dfresults = dfresults.append(dfresultappend)


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

print(seg_feat_house.shape)




    ##############################################################################################################
    #    code for saving segmented images (car, house, road, tree, terrain etc)
    ##############################################################################################################

    import numpy as np
    import cv2
    import sys

    caffe_root = '/home/student/Documents/PSPNet'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    sys.path.insert(0, caffe_root + 'python/')

    import caffe


    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # caffe.set_mode_cpu()
    model_def = '/home/cv/Sumukh/PSPNet/evaluation/prototxt/pspnet101_cityscapes_713.prototxt'
    model_weights = '/home/cv/Sumukh/PSPNet/caffemodel/pspnet101_cityscapes.caffemodel'

    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    mu = np.load('/home/cv/Sumukh/PSPNet/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    # mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    net.blobs['data'].reshape(1,  # batch size
                              3,  # 3-channel (BGR) images
                              713, 713)  # image size is 227x227
    for i in range(5653, 5654): #5464
        filename = 'file' + str(i) + '.png'
        image = caffe.io.load_image('/home/cv/Sumukh/Living_Indicator/img/' + filename)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()

        img1 = net.blobs['conv6_interp'].data[0, 0]  # road
        img2 = net.blobs['conv6_interp'].data[0, 2]  # house
        img3 = net.blobs['conv6_interp'].data[0, 8]  # trees
        img4 = net.blobs['conv6_interp'].data[0, 9]  # terrain
        img5 = net.blobs['conv6_interp'].data[0, 13]  # Car

        ret, th1 = cv2.threshold(img1, 6, 255, cv2.THRESH_BINARY)
        ret, th2 = cv2.threshold(img2, 9, 255, cv2.THRESH_BINARY)
        ret, th3 = cv2.threshold(img3, 6, 255, cv2.THRESH_BINARY)
        ret, th4 = cv2.threshold(img4, 5, 255, cv2.THRESH_BINARY)
        ret, th5 = cv2.threshold(img5, 5, 255, cv2.THRESH_BINARY)

        plt.imshow(th1)
        plt.savefig('/home/cv/Sumukh/Results/Road/test_road' + str(i) + '.png')
        plt.imshow(th2)
        plt.savefig('/home/cv/Sumukh/Results/House/test_house' + str(i) + '.png')
        plt.imshow(th3)
        plt.savefig('/home/cv/Sumukh/Results/Trees/test_tree' + str(i) + '.png')
        plt.imshow(th4)
        plt.savefig('/home/cv/Sumukh/Results/Terrain/test_terrain' + str(i) + '.png')
        plt.imshow(th5)
        plt.savefig('/home/cv/Sumukh/Results/Car/test_car' + str(i) + '.png')

# def get_Living_Index(res):
#     original_features = []
#     for i in range(0, len(res)):
#         try:
#             tax_value = float(res[i].tax_value)
#         except:
#             tax_value = 500000.0
#
#         try:
#             year_built = float(res[i].year_built)
#         except:
#             year_built = 1990.0
#
#         try:
#             property_size = float(res[i].property_size)
#         except:
#             property_size = 1000.0
#
#         try:
#             home_size = float(res[i].home_size)
#         except:
#             home_size = 1000.0
#
#         try:
#             bathrooms = float(res[i].bathrooms)
#         except:
#             bathrooms = 2.0
#
#         try:
#             bedrooms = float(res[i].bedrooms)
#         except:
#             bedrooms = 3.0
#
#         try:
#             zestimate_amount = float(res[i].zestimate_amount)
#         except:
#             zestimate_amount = 500000.0
#
#         try:
#             last_sold_price = float(res[i].last_sold_price)
#         except:
#             last_sold_price = 500000.0
#
#         # original_features.append([tax_value, property_size / home_size, bathrooms, bedrooms])  # year_built,property_size,zestimate_amount
#         original_features.append([property_size / home_size,
#                               last_sold_price])  # tax_value,year_built,property_size, bathrooms, bedrooms, zestimate_amount, last_sold_price
#
#     arr_orig_features = np.array(original_features)
#     for i in arr_orig_features:
#         if (i[1] >= 600000):
#             i[1] = 600000
#
#     score_pca = PCA(n_components=1)
#     score = score_pca.fit_transform(arr_orig_features)
#
#     finalscore = []
#     for i in score:
#         val = 0 + ((i[0] - min(score)) * 10) / (max(score) - min(score))
#         val = np.round(val)
#         # val = int(val)
#         finalscore.append(int(np.asscalar(val)))
#     return finalscore