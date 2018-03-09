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
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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