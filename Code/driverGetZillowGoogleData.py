#########################################################################################
# Download Zillow and Google Street view Data (Calls for the Get_Zillow_data() function
#----------------------------------------------------------------------------------------
#with open('LA.csv', 'rU') as infile:
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
