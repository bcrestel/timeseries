import numpy as np

def kalmanfilter(dataset, m0, C0, Ft, Gt, V, W):
    """ Compute Kalman filter on dataset 
    starting with initial distribution N(m0, C0).
    Inputs:
        dataset = np.array containing data -- (time steps) x (observations)
        m0 = mean of initial state
        C0 = covariance matrix of initial state
        F, G = matrices defining the DLM
        V, W = covariance matrices for observation and model
    Outputs:
        m_all = means of state estimate (theta_t | y_{1:t})
        C_all = covariance matrices of state estimate
        a_all = means for state predictive (theta_t | y_{1:t-1})
        R_all = covariance matrices for state predictive """
    timesteps = len(dataset)
    param = m0.size
    m = m0.reshape((param, 1))
    C = C0
    m_all = np.zeros((timesteps, param))
    a_all = np.zeros((timesteps, param))
    C_all, R_all = [], []
    ii = 0
    for YT, F, G in zip(dataset, Ft, Gt):
        Y = YT.T
        # State predictive
        a = G.dot(m)
        R = W + G.dot(C).dot(G.T)
        a_all[ii,:] = a.T
        R_all.append(R)
        # Intermediate step
        e = Y - F.dot(a)
        Q = V + F.dot(R).dot(F.T)
        Qinv = np.linalg.inv(Q)
        KGain = R.dot(F.T).dot(Qinv)
        # State estimate
        m = a + KGain.dot(e)
        C = R - KGain.dot(F).dot(R)
        m_all[ii,:] = m.T
        C_all.append(C)
        ii += 1
    return m_all, C_all, a_all, R_all
