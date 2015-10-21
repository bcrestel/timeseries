import numpy as np
from numpy.linalg import cholesky

def KalmanFilter(dataset, m0, C0, Ft, Gt, V, W):
    """ Compute Kalman filter on dataset 
    starting with initial distribution N(m0, C0).
    Inputs:
        dataset = np.array containing data -- (time steps) x (observations)
        m0 = mean of initial state
        C0 = covariance matrix of initial state
        Ft, Gt = list of matrices defining the DLM
        V, W = covariance matrices for observation and model
    Outputs:
        m_all = means of state estimate (theta_t | y_{1:t})
        C_all = covariance matrices of state estimate
        a_all = means for state predictive (theta_t | y_{1:t-1})
        R_all = covariance matrices for state predictive """
    timesteps = len(dataset)
    nbobs = dataset.size/timesteps
    param = m0.size
    m = m0.reshape((param, 1))
    C = C0
    m_all = np.zeros((timesteps, param))
    a_all = np.zeros((timesteps, param))
    C_all, R_all = [], []
    ii = 0
    for YT, F, G in zip(dataset, Ft, Gt):
        Y = YT.reshape((nbobs,1))
        # State predictive
        a = G.dot(m)
        Csq = cholesky(C)
        GCsq = G.dot(Csq)
        R = W + GCsq.dot(GCsq.T)
        a_all[ii,:] = a.T
        R_all.append(R)
        # Intermediate step
        e = Y - F.dot(a)
        Rsq = cholesky(R)
        FRsq = F.dot(Rsq)
        Q = V + FRsq.dot(FRsq.T)
        Qinv = np.linalg.inv(Q)
        Qinvsq = cholesky(Qinv)
        # State estimate
        RFt = R.dot(F.T)
        RFtQsq = RFt.dot(Qinvsq)
        m = a + RFt.dot(Qinv.dot(e))
        C = R - RFtQsq.dot(RFtQsq.T)
        m_all[ii,:] = m.T
        C_all.append(C)
        ii += 1
    return m_all, C_all, a_all, R_all


def KalmanSmoother(sT, ST, Gt, m_all, C_all, a_all, R_all):
    """ Compute Kalman smoother from output of Kalman filter
    Inputs:
        sT, ST = mean and covariance of (theta_T | y_{1:T})
        Gt = list of matrices containing definition of DLM
    Outputs:
        s_all = mean of (theta_t | y_{1:T}), (time steps) x (parameters)
        S_all = list of covariance matrices of (theta_t | y_{1:T}) """
    timesteps, parameters = m_all.shape
    s_all, S_all = np.zeros(m_all.shape), []
    s_all[timesteps-1,:] = sT
    S_all.append(ST)
    s, S = sT.T, ST
    # Iterate starting from t=T down to t=1:
    ii = timesteps - 2
    for G, R in zip(reversed(Gt), reversed(R_all)):
        a, m, C = a_all[ii+1,:].T, m_all[ii,:].T, C_all[ii]
        invR = np.linalg.inv(R)
        CGR = C.dot(G.T).dot(invR)
        CGRt = invR.dot(G).dot(C)
        s = m + CGR.dot(s - a)
        S = C - CGR.dot(R - S).dot(CGR.T)
        #S = C - CGR.dot((R - S).dot(CGRt))
        s_all[ii,:] = s.T
        S_all.append(S)
        ii -= 1
        if ii < 0: break
    return s_all, S_all[::-1]
