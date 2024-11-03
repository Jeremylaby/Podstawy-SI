from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
import numpy as np


def assess_regression_model(model, X_train, X_test, y_train, y_test) -> tuple[float, float]:
    # predict for train and test
    # your_code_here
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    # exponential transform for y_train, y_test and predictions
    # your_code_here
    y_train_predict = np.expm1(y_train_predict)
    y_test_predict = np.expm1(y_test_predict)
    y_train = np.expm1(y_train)
    y_test = np.expm1(y_test)

    # calculate train and test RMSE
    # your_code_here
    rmse_train = root_mean_squared_error(y_train, y_train_predict)
    rmse_test = root_mean_squared_error(y_test, y_test_predict)

    # print train and test RMSE
    # your_code_here
    print(f"RMSE (train): {rmse_train:.2f}")
    print(f"RMSE (test): {rmse_test:.2f}")

    # your_code
    return rmse_train, rmse_test


reg_ridge = Ridge(random_state=0)
reg_lasso = Lasso(random_state=0)

reg_ridge = RidgeCV(alphas=(np.linspace(0.1, 100, 1000)), random_state=0)
reg_lasso = LassoCV(n_alphas=1000, cv=5, random_state=0)
reg_ridge.fit(X_train, y_train)
reg_lasso.fit(X_train, y_train)

assess_regression_model(reg_ridge, X_train, X_test, y_train, y_test)
print()
assess_regression_model(reg_lasso, X_train, X_test, y_train, y_test)
print()
