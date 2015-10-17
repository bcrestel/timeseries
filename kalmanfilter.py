import numpy as np

def kalmanfilter(dataset, m0, C0, F, G, V, W):
    """ Compute Kalman filter on dataset 
    starting with initial distribution N(m0, C0).
    Inputs:
        dataset = np.array containing data -- (observations) x (time steps)
        m0 = mean of initial state
        C0 = covariance matrix of initial state
        F, G = matrices defining the DLM
        V, W = covariance matrices for observation and model
    Outputs:
        m_all = all successive means for state
        C_all = all successive covariance matrices at each time step
        a_all = all successive means for state predictive 
        R_all = all successive covariance matrices at each time step """
    m_all, a_all, C_all, R_all = [], [], [], []
    nbobs, timesteps = dataset.shape
    m = m0
    C = C0
    for ii in range(timesteps):
        Y = dataset[:,ii]
        # State predictive
        a = G.dot(m)
        R = W + G.dot(C).dot(G.T)
        a_all.append(a)
        R_all.append(R)
        # Intermediate step
        e = Y - F.dot(a)
        Q = V + F.dot(R).dot(F.T)
        Qinv = np.linalg.inv(Q)
        KGain = R.dot(F.T).dot(Qinv)
        # State estimate
        m = a + KGain.dot(e)
        C = R - KGain.dot(F).dot(R)
        m_all.append(m)
        C_all.append(C)
    return m_all, C_all, a_all, R_all
