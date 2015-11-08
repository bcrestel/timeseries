import numpy as np
from numpy.linalg import cholesky, svd

from fenicstools.sampling.distributions import SampleMultivariateNormal


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


def KalmanFilter_SVD(dataset, m0, svd_C0, Ft, Gt, svd_invV, svd_invW):
    """ Compute Kalman filter on dataset 
    starting with initial distribution N(m0, C0).
    Inputs:
        dataset = np.array containing data -- (time steps) x (observations)
        m0 = mean of initial state
        svd_C0 = [U,S] for covariance matrix of initial state
        Ft, Gt = list of matrices defining the DLM
        svd_invV = SVD factors of the inverse of V, i.e., [U, S] such that
            V^{-1} = U.S.U^T
        svd_invV = same as for invV but with W
        (V, W = covariance matrices for observation and model)
    Outputs:
        m_all = means of state estimate (theta_t | y_{1:t})
        C_all = covariance matrices of state estimate
        a_all = means for state predictive (theta_t | y_{1:t-1})
        R_all = covariance matrices for state predictive """
    timesteps = len(dataset)
    nbobs = dataset.size/timesteps
    param = m0.size
    m = m0.reshape((param, 1))
    C = [svd_C0[0], np.sqrt(svd_C0[1])]
    Gam = svd_invW[0].dot(np.diag(1/np.sqrt(svd_invW[1])))
    invV_ch = svd_invV[0].dot(np.diag(np.sqrt(svd_invV[1])))
    V = svd_invV[0].dot(np.diag(1/svd_invV[1])).dot(svd_invV[0].T)
    m_all = np.zeros((timesteps, param))
    a_all = np.zeros((timesteps, param))
    C_all, R_all = [], []
    ii = 0
    for YT, F, G in zip(dataset, Ft, Gt):
        Y = YT.reshape((nbobs,1))
        # State predictive: a, R
        a = G.dot(m)
        a_all[ii,:] = a.T
        tmp = np.diag(C[1]).dot((C[0].T).dot(G.T))
        Z, D, UT = svd(np.concatenate((tmp, Gam.T), axis=0))
        U = UT.T
        R_all.append([U, D])
        # Intermediate step: e, Q
        e = Y - F.dot(a)
        FRsq = F.dot(U.dot(np.diag(D)))
        Q = V + FRsq.dot(FRsq.T)
        Qinve = np.linalg.solve(Q, e)
        # State estimate: m, C
        m = a + U.dot(np.diag(D**2).dot(UT)).dot(F.T).dot(Qinve)
        m_all[ii,:] = m.T
        tmp = (invV_ch.T).dot(F.dot(U))
        Delta, tDinv, MT = svd(np.concatenate((tmp, np.diag(1/D)), axis=0))
        M = MT.T
        C = [U.dot(M), 1/tDinv]
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
        s_all[ii,:] = s.T
        S_all.append(S)
        ii -= 1
        if ii < 0: break
    return s_all, S_all[::-1]


def BackwardSampling_SVD(Gt, svd_invW, m0, m_all, svd_C0, C_all, a_all, R_all):
    """ Sample from the joint distribution (theta_{0:T} | y_{1:T})
    Inputs:
        Gt = list of matrices containing definition of DLM
        svd_invW = [U, S] such that U.S.U^T = W^{-1}
        (W = covariance matrix for evolution equation)
        m0, C0 = parameters of initial distribution
        m_all, C_all, a_all, R_all = output from KalmanFilter_SVD
    Outputs:
        thetas = joint draw from (theta_{0:T} | y_{1:T}) 
        maxVarB = maximum variance in B at each time step (diagonal only) """
    timesteps, parameters = m_all.shape
    thetas = np.zeros((timesteps+1, parameters))
    invGam = svd_invW[0].dot(np.diag(np.sqrt(svd_invW[1])))
    # Sample at time t=T
    h = m_all[timesteps-1,:].reshape((parameters,1))
    C = C_all[-1]
    B_ch = C[0].dot(np.diag(C[1]))
    theta = SampleMultivariateNormal(h, B_ch)
    thetas[-1,:] = theta.reshape((1, parameters))
    maxVarB = []
    # To check, compute sT:
#    s_all, S_all = np.zeros(m_all.shape), []
#    s = m_all[timesteps-1,:].reshape((parameters,1))
#    s_all[timesteps-1,:] = s.T
#    Sfull = C[0].dot(np.diag(C[1]**2)).dot(C[0].T)
#    S_all.append(Sfull)
    # Iterate starting from t=T down to t=1:
    ii = timesteps - 2
    for G, Rsvd in zip(reversed(Gt), reversed(R_all)):
        a = a_all[ii+1,:].reshape((parameters,1))
        if ii < 0: 
            m = m0.reshape((parameters,1))
            C = svd_C0[0].dot(np.diag(svd_C0[1])).dot(svd_C0[0].T)
            invR = Rsvd[0].dot(np.diag(1/Rsvd[1]**2)).dot(Rsvd[0].T)
            CGinvR = C.dot(G.T).dot(invR)
            h = m + CGinvR.dot(theta - a)
            tmp = (invGam.T).dot(G).dot(svd_C0[0])
            A, invD, ET = svd(np.concatenate((tmp, \
            np.diag(1/np.sqrt(svd_C0[1]))), axis=0))
            U = svd_C0[0].dot(ET.T)
            B_ch = U.dot(np.diag(1/invD))
            theta = SampleMultivariateNormal(h, B_ch)
            thetas[ii+1,:] = theta.reshape((1, parameters))
            maxVarB.append(np.abs(np.diag(B_ch.dot(B_ch.T))).max())
            break
        m = m_all[ii,:].reshape((parameters,1))
        Csvd = C_all[ii]
        C = Csvd[0].dot(np.diag(Csvd[1]**2)).dot(Csvd[0].T)
        invR = Rsvd[0].dot(np.diag(1/Rsvd[1]**2)).dot(Rsvd[0].T)
        CGinvR = C.dot(G.T).dot(invR)
        h = m + CGinvR.dot(theta - a)
        tmp = (invGam.T).dot(G).dot(Csvd[0])
        A, invD, ET = svd(np.concatenate((tmp, np.diag(1/Csvd[1])), axis=0))
        U = Csvd[0].dot(ET.T)
        B_ch = U.dot(np.diag(1/invD))
        theta = SampleMultivariateNormal(h, B_ch)
        thetas[ii+1,:] = theta.reshape((1, parameters))
        maxVarB.append(np.abs(np.diag(B_ch.dot(B_ch.T))).max())
        # To check, compute st
#        s = m + CGinvR.dot(s - a)
#        s_all[ii,:] = s.T
#        Bfull = C - CGinvR.dot(G).dot(C)
#        Bfull = C - C.dot(G.T).dot(invR).dot(G).dot(C)
#        invBfullbis = np.linalg.inv(C) + \
#        (G.T).dot(np.linalg.inv(W)).dot(G)
#        Bfullbis = np.linalg.inv(invBfullbis)
#        Rfull = Rsvd[0].dot(np.diag(Rsvd[1]**2)).dot(Rsvd[0].T)
#        Sfull = C - CGinvR.dot(Rfull-Sfull).dot(CGinvR.T)
#        S_all.append(Sfull)
        ii -= 1
    return thetas, maxVarB


def FFBS(dataset, m0, svd_C0, Ft, Gt, svd_invV, svd_invW):
    """ Sample from the joint distribution (theta_{1:T} | y_{1:T}) given
    covariance matrices V and W using the Forward Filtering Backward Sampling
    (FFBS) algorithm
    Inputs:
        dataset = np.array containing data -- (time steps) x (observations)
        m0 = mean of initial state
        svd_C0 = [U,S] for covariance matrix of initial state
        Ft, Gt = list of matrices defining the DLM
        Cholesky_invV = Cholesky factor of the inverse of V, i.e., [U, S] such that
            V^{-1} = U.S.U^T
        Cholesky_invV = same as for invV but with W
        (V, W = covariance matrices for observation and model)
    Outputs:
        thetas = joint draw from (theta_{1:T} | y_{1:T}) 
        maxVarB = maximum variance in B at each time step (diagonal only) """
    m, C, a, R = KalmanFilter_SVD(dataset, m0, svd_C0, Ft, Gt, svd_invV, svd_invW)
    return BackwardSampling_SVD(Gt, svd_invW, m0, m, svd_C0, C, a, R)
    




######################
###### VOID ##########
def KalmanFilter2(dataset, m0, C0, Ft, Gt, V, W):
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
    Us,Ss,Vst = svd(C0)
    Cs = [Us,np.sqrt(Ss)]
    a,s,b = svd(W)
    Gam = a.dot(np.diag(np.sqrt(s)))
    S = cholesky(np.linalg.inv(V))
    m_all = np.zeros((timesteps, param))
    a_all = np.zeros((timesteps, param))
    C_all, R_all = [], []
    ii = 0
    for YT, F, G in zip(dataset, Ft, Gt):
        Y = YT.reshape((nbobs,1))
        #print "C:", np.linalg.norm(C - \
        #Cs[0].dot(np.diag(Cs[1]**2).dot(Cs[0].T)))/np.linalg.norm(C)
        # State predictive
        a = G.dot(m)
        Csq = cholesky(C)
        #print "Csq:", np.linalg.norm(Csq - Cs[0].dot(np.diag(Cs[1])))\
        #/np.linalg.norm(Csq)
        GCsq = G.dot(Csq)
        R = W + GCsq.dot(GCsq.T)
        a_all[ii,:] = a.T
        R_all.append(R)

        tmp = np.diag(Cs[1]).dot((Cs[0].T).dot(G.T))
        Z, D, UT = svd(np.concatenate((tmp, Gam.T), axis=0))
        U = UT.T
        print "R:", np.linalg.norm(R - \
        U.dot(np.diag(D**2).dot(U.T)))/np.linalg.norm(R)

        UR, DR, VTR = svd(R)
        print "D:", np.linalg.norm(D**2-DR)/np.linalg.norm(DR)
        # Intermediate step
        e = Y - F.dot(a)
        #FRsq2 = F.dot(U.dot(np.diag(D)))
        #Q2 = V + FRsq2.dot(FRsq2.T)
        #Qinve2 = np.linalg.solve(Q2, e)
        #
        Rsq = cholesky(R)
        FRsq = F.dot(Rsq)
        Q = V + FRsq.dot(FRsq.T)
        #print "Q:", np.linalg.norm(Q-Q2)/np.linalg.norm(Q)
        Qinv = np.linalg.inv(Q)
        Qinvsq = cholesky(Qinv)
        #print "inv(Q).e:", np.linalg.norm(Qinve2 - Qinv.dot(e))
        # State estimate
        #m2 = a + U.dot(np.diag(D**2).dot(UT)).dot(F.T).dot(Qinve2)
        RFt = R.dot(F.T)
        RFtQsq = RFt.dot(Qinvsq)
        m = a + RFt.dot(Qinv.dot(e))
        #print "m:", np.linalg.norm(m-m2)/np.linalg.norm(m)
        C = R - RFtQsq.dot(RFtQsq.T)
        m_all[ii,:] = m.T
        C_all.append(C)
        #
        #tmp = (S.T).dot(F.dot(U))
        #Delta, tDinv, MT = svd(np.concatenate((tmp, np.diag(1/D)), axis=0))
        #M = MT.T
        #tU = U.dot(M)
        #tD = 1/tDinv
        #Cs = [tU,tD]
        tU, tD, tVT = svd(C)
        Cs = [tU, np.sqrt(tD)]
        print "check svd(C):", np.linalg.norm(C - \
        Cs[0].dot(np.diag(Cs[1]**2).dot(Cs[0].T)))/np.linalg.norm(C)
        #print tD**2, "\n", tD2
        #print "tD:", np.linalg.norm(tD2-tD**2)/np.linalg.norm(tD2), "\n"
        ii += 1
        if ii == 2: break
        print 
    return m_all, C_all, a_all, R_all
