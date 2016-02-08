import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from timeseries.dlm import KalmanFilter, KalmanSmoother, FFBS
from fenicstools.sampling.distributions import Wishart_fromSigmaInverse, \
SampleMultivariateNormal

""" Reproduce multivariate CAPM example from section 3.3.3 in Petris et al. 
The idea of the CAPM model is to compute a dynamic linear regression where the
observations are the excess return of an asset and the regressors are the
markets' excess return (it seems intuitive that the excess returns correspond to
the discounted variables in continuous-time finance.). 

Parameterization:
We have m stocks and the parameterization is done as
theta = [a_1, .... ,a_m, b_1, ..., b_m] """

DEBUG = True
if DEBUG:   SAVEFILE = False
else:   SAVEFILE = True

# Note this importance of the scaling factor of 100 in front.
# One gets completely different results w/o it.
dataset = 100.*np.genfromtxt('stocksIBMandCo.txt', skip_header=1)
timesteps, col = dataset.shape
T = timesteps
riskfree = dataset[:,6].reshape((timesteps,1))
# excess return of assets (= asset - risk_free):
Y = dataset[:,1:5] - riskfree*np.array([[1,1,1,1]])
# excess return of the market (= Market = risk_free):
X = dataset[:,5] - dataset[:,6]
tmp, nbstock = Y.shape
Ft = [np.concatenate((np.eye(nbstock), x*np.eye(nbstock)), axis=1) for x in X]
Gt = [np.eye(2*nbstock) for ii in range(len(X))]
# This values of V and W are coming from the example 3.3.3 in Petris et al.
# (computed with MLE)
V = np.array([[41.06,0.01571,-0.9504,-2.328], [0.01571,24.23,5.783,3.376],
[-0.9504,5.783,39.2,8.145], [-2.328,3.376,8.145,39.29]])
W = np.zeros((2*nbstock, 2*nbstock))
W[nbstock:, nbstock:] = np.array([[8.153e-7,-3.172e-5,-4.267e-5,-6.649e-5],
[-3.172e-5,0.001377,0.001852,0.002884], [-4.267e-5,0.001852,0.002498,0.003884],
[-6.649e-5,0.002884,0.003884,0.006057]])
W[:nbstock, :nbstock] = 1e-12*np.diag(np.ones(nbstock))
m0 = np.zeros((2*nbstock,1))
C0 = 1e7*np.eye(2*nbstock)


# Compute simple smoothing
m, C, a, R = KalmanFilter(Y, m0, C0, Ft, Gt, V, W)
s, S = KalmanSmoother(m[-1,:], C[-1], Gt, m0, C0, m, C, a, R)

# W = [W1|0][0|W2]. Assume V and W1 are known and not part of the inference.
W2_priormean = np.diag(1./np.diag(np.linalg.inv(W[nbstock:,nbstock:])))
#W2_priormean = W[nbstock:,nbstock:]
try:
    mu0 = float(sys.argv[1])
except:
    mu0 = nbstock+1
invT0 = mu0*W2_priormean #T0 = np.linalg.inv(W)/mu0
# Compute initial svd decompositions:
Uc, Sc, UcT = np.linalg.svd(C0)
svd_C0 = [Uc, Sc]
Uv, Sv, UvT = np.linalg.svd(V)
svd_invV = [Uv, 1/Sv]
Uw1, Sw1, Uw1T = np.linalg.svd(W[:nbstock,:nbstock])
Uw2, Sw2, Uw2T = np.linalg.svd(np.eye(nbstock))
def assembleSVDinvW(Uw1, Sw1, Uw2, Sw2):
    """ Inputs are for the SVD of the INVERSE of W1 and W2 """
    Z = np.zeros((nbstock,nbstock))
    W1Z = np.concatenate((Uw1,Z), axis=0)
    W2Z = np.concatenate((Z,Uw2), axis=0)
    Uw = np.concatenate((W1Z,W2Z), axis=1)
    Sw = np.concatenate((Sw1,Sw2))
    return [Uw, Sw]
svd_invW = assembleSVDinvW(Uw1, 1/Sw1, Uw2, 1/Sw2)
if DEBUG:   
    print Uw2.dot(np.diag(Sw2)).dot(Uw2.T)
    print Uw2.dot(np.diag(1/Sw2)).dot(Uw2.T)
sample_size = 100000
AllThetas = np.zeros((T+1, 2*nbstock, sample_size))
AllinvW2 = np.zeros((nbstock, nbstock, sample_size))
if DEBUG:
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax.plot(s[:,4], 'k')
    ax2.plot(s[:,5], 'k')
    ax3.plot(s[:,6], 'k')
    ax4.plot(s[:,7], 'k')
    line1, = ax.plot(s[:,4], 'b')
    line2, = ax2.plot(s[:,5], 'g')
    line3, = ax3.plot(s[:,6], 'r')
    line4, = ax4.plot(s[:,7], 'c')
    ax.set_ylim(0.,2.)
    ax.set_title('MOBIL')
    ax2.set_ylim(0.,2.)
    ax2.set_title('IBM')
    ax3.set_ylim(0.,2.)
    ax3.set_title('WEYER')
    ax4.set_ylim(0.,2.)
    ax4.set_title('CITCRP')
    plt.show(block=False)
for ii in range(sample_size):
    # Sample from (theta_{0:T} | Y_{1:T}, V, W)
    thetas, tmp = FFBS(Y, m0, svd_C0, Ft, Gt, svd_invV, svd_invW)
    AllThetas[:,:,ii] = thetas
    if DEBUG:
        line1.set_ydata(thetas[:,4])
        line2.set_ydata(thetas[:,5])
        line3.set_ydata(thetas[:,6])
        line4.set_ydata(thetas[:,7])
        plt.pause(.01)
    betas_i = thetas[1:,nbstock:] # only keep the betas
    betas_iminusone = thetas[:-1,nbstock:]
    # Sample from (W2 | theta_{0:T}, Y_{1:T})
    Gb = np.zeros((timesteps, nbstock))
    for jj, (Gfull, b_im) in enumerate(zip(Gt, betas_iminusone)):
        G = Gfull[nbstock:,nbstock:]
        Gb[jj,:] = (G.dot(b_im.reshape((b_im.size,1)))).T
    bmGbi = (betas_i - Gb).T
    SigmaInv = invT0 + bmGbi.dot(bmGbi.T)
    if DEBUG:   
        print '\n', invT0
        print bmGbi.dot(bmGbi.T)
        print np.linalg.inv(SigmaInv)
    Usig, Ssig, UsigT = np.linalg.svd(SigmaInv)
    invW2 = Wishart_fromSigmaInverse(mu0+T, SigmaInv, 1e-11)
    if DEBUG:   print invW2
    # Update V and W for next step
    Uw2, Sw2, Uw2T = np.linalg.svd(invW2)
    svd_invW = assembleSVDinvW(Uw1, 1/Sw1, Uw2, Sw2)
    AllinvW2[:,:,ii] = invW2
    if DEBUG and ii%500== 0:   
        print 'Iteration {0}'.format(ii)
        tmp = raw_input("Press <ENTER> to continue")
# Save file if required:
if SAVEFILE == True:    
    filename, extension = os.path.splitext(sys.argv[0])
    namefile = 'Outputs/' + filename + \
    '-samplesize' + str(sample_size) + \
    '-mu0' + str(int(mu0))
    np.savez_compressed(namefile, AllThetas, AllinvW2)

