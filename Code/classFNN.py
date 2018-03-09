def neural_net(trainx, trainy, testx, testy):
    model = MLPClassifier(activation='tanh', solver ='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(trainx[0]),2), random_state=1)
    model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy