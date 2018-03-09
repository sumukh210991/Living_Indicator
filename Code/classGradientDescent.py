
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