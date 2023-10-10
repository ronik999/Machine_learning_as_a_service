from sklearn.tree import DecisionTreeRegressor

def train_predictive_model(X_train, y_train, X_test, y_test):
    '''
    Input Parameters:
    -----------------
    X_train: Splitted train features for predictors (X)
    y_train: Splitted train revenues (y)
    X_test: Splitted test features for predictors (X)
    y_test: Splitted test revenues (y)

    Output:
    ________
    model: Trained model

    '''
    model = DecisionTreeRegressor(max_depth=15,
                                  min_samples_split=5,
                                  min_samples_leaf=6,
                                  max_features=None
                                  )
    model.fit(X_train, y_train)
    return model
