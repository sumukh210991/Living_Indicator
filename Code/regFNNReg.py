def nn_regressor(X_train, y_train, X_test):
    clf = MLPRegressor(hidden_layer_sizes=(70, 20), activation='relu', solver='sgd', alpha=0.000001,
                       learning_rate='adaptive', max_iter=500, random_state=0, tol=1e-5) # use (100, 50) for Sampling col_hist
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred