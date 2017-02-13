# import pyzillow
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults
import csv
import requests
import urllib.parse, urllib.request
import os
from sklearn.decomposition import PCA
import numpy
from matplotlib import pyplot as plt
import cv2

def get_zillow_data(addr_list, zip_list):
    meta = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&"
    myloc = "C:\\Users\\sumukh210991\\Desktop\\Study\\Thesis\\popularity_score_for_houses\\python\\img"
    key = "&key=" + "AIzaSyC7-APuKb-aknoKymJdflh2jTC91HBe8rY"
    data = []
    count = 0 # 721
    for i in range(0, len(addr_list)):
        address = addr_list[i]
        zipcode = zip_list[i]
        zillow_data = ZillowWrapper("X1-ZWz1fijmr1w9hn_9fxlx")  # nishant_masc 'X1-ZWz19gfhe5z8y3_8xo7s'
        # nishant:'X1-ZWz1fk2vm7fkln_8tgid';
        # sumukh: 'X1-ZWz1fijmr1w9hn_9fxlx';
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
            metaurl = meta + urllib.parse.urlencode(params)
            MyUrl = base + urllib.parse.urlencode(params)
            file_loc = "img/file" + (str(count)) + ".png"
            print(file_loc)
            status = requests.get(metaurl)
            if ("ZERO_RESULTS" in str(status.content)):
                print("NO CONTENT")
                continue
            else:
                count = count + 1
                res = urllib.request.urlretrieve(MyUrl, os.path.join(file_loc))
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
    metaurl = meta + urllib.parse.urlencode(params)
    MyUrl = base + urllib.parse.urlencode(params)
    #fi = SaveLoc + r"\myfile.png"
    status = requests.get(metaurl)
    if ("ZERO_RESULTS" in str(status.content)):
        print("TRUE")
    else:
        print("FALSE")
    print(status.content)
    print(status)
    res = urllib.request.urlretrieve(MyUrl, os.path.join(fi))
    print(res)
    return res


def get_Living_Index(res):
    original_features = []
    for i in range(0, len(res)):
        try:
            tax_value = float(res[i].tax_value)
        except:
            tax_value = 500000.0

        try:
            year_built = float(res[i].year_built)
        except:
            year_built = 1990.0

        try:
            property_size = float(res[i].property_size)
        except:
            property_size = 1000.0

        try:
            home_size = float(res[i].home_size)
        except:
            home_size = 1000.0

        try:
            bathrooms = float(res[i].bathrooms)
        except:
            bathrooms = 2.0

        try:
            bedrooms = float(res[i].bedrooms)
        except:
            bedrooms = 3.0

        try:
            zestimate_amount = float(res[i].zestimate_amount)
        except:
            zestimate_amount = 500000.0

        try:
            last_sold_price = float(res[i].last_sold_price)
        except:
            last_sold_price = 500000.0

        # original_features.append([tax_value, property_size / home_size, bathrooms, bedrooms])  # year_built,property_size,zestimate_amount
        original_features.append([property_size / home_size,
                              last_sold_price])  # tax_value,year_built,property_size, bathrooms, bedrooms, zestimate_amount, last_sold_price

    arr_orig_features = numpy.array(original_features)
    for i in arr_orig_features:
        if (i[1] >= 600000):
            i[1] = 600000

    score_pca = PCA(n_components=1)
    score = score_pca.fit_transform(arr_orig_features)

    finalscore = []
    for i in score:
        val = 0 + ((i[0] - min(score)) * 10) / (max(score) - min(score))
        val = numpy.round(val)
        # val = int(val)
        finalscore.append(int(numpy.asscalar(val)))
    return finalscore


def get_colhist(filenames):
    #features = []
    colhist = []
    for names in filenames:
        image = cv2.imread(names)
        chans = cv2.split(image)
        #colhist = []
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
            hist = cv2.normalize(numpy.array(hist), dst=cv2.NORM_MINMAX)
            colhist.extend(hist)
        #features.append(numpy.array(colhist).flatten())
    features = numpy.array(numpy.array(colhist).flatten())
    features = features.reshape((len(filenames) - 2, 96))
    return features


def get_imgrad(filenames):
    #features = []
    gradhist = []
    for names in filenames:
        image = cv2.imread(names)
        chans = cv2.split(image)
        # gradhist = []
        for chan in chans:
            laplace = cv2.Laplacian(chan, cv2.CV_32F)
            #sobelx = cv2.Sobel(chan, cv2.CV_32F, 1, 0, ksize=5)
            #sobely = cv2.Sobel(chan, cv2.CV_32F, 0, 1, ksize=5)
            hist = cv2.calcHist([laplace], [0], None, [32], [0, 256]) #, sobely, laplace,
            hist = cv2.normalize(numpy.array(hist), dst=cv2.NORM_MINMAX)
            gradhist.extend(hist)
        #features.append(numpy.array(gradhist).flatten())
    feature = numpy.array(numpy.array(gradhist).flatten())
    feature = feature.reshape((len(filenames)-2, 96))
    return feature


def gradient_descent(alpha, x, y, numiter, epsilon):
    theta = numpy.zeros(len(x[0]))
    m = len(y)
    arrcost = []
    for _ in range(1,numiter):
        pred = numpy.dot(x, theta)
        temp = numpy.dot((pred - y), x)
        theta = theta - ((alpha / m) * temp)
        val = numpy.sum((numpy.dot(x,theta) - y) / m)
        arrcost.append(numpy.abs(val))
        if(numpy.abs(val) < epsilon):
            return {'cost': arrcost, 'theta': theta }
    return {'cost': arrcost, 'theta': theta}


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

# rand_idx = numpy.random.permutation(len(zip_all))

# random permutations saved in CSV for reuse in future : steps of 2000
numpy.savetxt("randperm.csv", rand_idx, delimiter= ",")

address = []
zip = []
for i in rand_idx[0:2000]:
    address.append(address_all[i])
    zip.append(zip_all[i])

res = get_zillow_data(address, zip) # 721, 1487, 2282


# USE THIS FOR ZILLOW RESULTS
#
# import pickle
#
# f = open('result.pickle', 'wb')
# pickle.dump(object, f)
# f.close()
#
f = open('result.pickle', 'rb')
res = pickle.load(f)
f.close()


living_index = get_Living_Index(res)

filenames = []
for i in range(0,len(res)+2):
    name = "img/file"+str(i)+".png"
    filenames.append(name)
feature = get_colhist(filenames)

#y = numpy.genfromtxt('pcascores.csv',delimiter = ',')
y = living_index
x = numpy.column_stack((numpy.ones(len(feature)), feature))

trainx = x[0:1800]
testx = x[1800:]
trainy = y[0:1800]
testy = y[1800:]
gd_res = gradient_descent(0.001, trainx, trainy, 1000000, 0.00001)

res = numpy.dot(testx, gd_res['theta'])
(len(res[numpy.round(res) == testy]) / len(testy)) * 100
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
# features.extend(numpy.array(colhist).flatten())