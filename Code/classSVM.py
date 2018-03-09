def Support_vector_class(trainx, trainy, testx, testy):
    model = svm.SVC(decision_function_shape='ovo')
    model.fit(trainx, trainy)
    res = model.predict(testx)
    accuracy = np.true_divide(len(res[np.round(res) == testy]), len(testy)) * 100
    return accuracy