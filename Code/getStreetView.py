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