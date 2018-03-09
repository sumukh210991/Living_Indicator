def decision_tree(trainx, trainy, testx, testy):
    features = 'auto'
    depth = 5
    model = tree.DecisionTreeClassifier(max_features = features, max_depth= depth)
    model = model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy