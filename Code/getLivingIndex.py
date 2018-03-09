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