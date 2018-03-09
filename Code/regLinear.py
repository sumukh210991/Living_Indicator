def lr_regression(X_train, y_train, X_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    # Calculate the R2, Std-Error in the driver code
    return y_pred