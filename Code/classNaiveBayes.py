def naive_bayes(trainx, trainy, testx, testy):
    model = GaussianNB()
    model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy