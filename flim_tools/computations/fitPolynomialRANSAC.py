import numpy as np
from sklearn.linear_model import RANSACRegressor


from sklearn.metrics import mean_squared_error


class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {"coeffs": self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


def fitPolynomialRANSAC(xyPoints, N, maxDistance):
    """
    Fit polynomial to points using RANSAC, returns best fit polynomial

    Parameters
    ----------
    xyPoints : m-by-2 matrix
        [x y] coordinate points
    N : int
        Degree of polynomial fit
    maxDistance : positive scalar
        Maximum distance for inlier points

    Returns ransac object that can be used to fit a polynomial
    -------

    """
    # x_vals, y_vals = xyPoints

    ransac = RANSACRegressor(
        PolynomialRegression(degree=N), maxDistance, random_state=0
    )

    return ransac
