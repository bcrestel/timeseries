import numpy as np

def InferRegress_NormalNoise(Y, X, sigma2, m0, C0):
    """ Inference on the regression coefficients of the form: Y = X.b + e,
    assuming noise covariance is know.
    Noise is e ~ N(0, sigma2.I), 
    prior is p(b) ~ N(m0, C0)
    and posterior is p(b|Y,sigma2) ~ N(m1, C1) """
    invC0 = np.linalg.inv(C0)
    C1 = np.linalg.inv(invC0 + (X.T).dot(X)/sigma2)
    m1 = C1.dot(invC0.dot(m0) + (X.T).dot(Y)/sigma2)
    return m1, C1
