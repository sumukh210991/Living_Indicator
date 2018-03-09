
def rf_regressor(X_train, y_train, X_test):
    regr = RandomForestRegressor(n_estimators=500, criterion='mse', max_features=20, max_depth=5, min_samples_split=200,
                                 bootstrap=True, oob_score=True, random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    return y_pred