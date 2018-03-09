def bayesian_regression(X_train, y_train, X_test):
    clf = linear_model.BayesianRidge()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate the R2, Std-Error in the driver code
    return y_pred