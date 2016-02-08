import numpy as np
import matplotlib.pyplot as plt

from timeseries.dlm import KalmanFilter, KalmanSmoother, FFBS

""" Reproduce multivariate CAPM example from section 3.3.3 in Petris et al. 
The idea of the CAPM model is to compute a dynamic linear regression where the
observations are the excess return of an asset and the regressors are the
markets' excess return (it seems intuitive that the excess returns correspond to
the discounted variables in continuous-time finance.). 

Parameterization:
We have m stocks and the parameterization is done as
theta = [a_1, .... ,a_m, b_1, ..., b_m] """

# Note this importance of the scaling factor of 100 in front.
# One gets completely different results w/o it.
dataset = 100.*np.genfromtxt('stocksIBMandCo.txt', skip_header=1)
timesteps, col = dataset.shape
riskfree = dataset[:,6].reshape((timesteps,1))
# excess return of assets (= asset - risk_free):
Y = dataset[:,1:5] - riskfree*np.array([[1,1,1,1]])
# excess return of the market (= Market = risk_free):
X = dataset[:,5] - dataset[:,6]
tmp, nbstock = Y.shape
Ft = [np.concatenate((np.eye(nbstock), x*np.eye(nbstock)), axis=1) for x in X]
Gt = [np.eye(2*nbstock) for ii in range(len(X))]
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
fig1 = plt.figure()
axx = fig1.add_subplot(111)
axx.plot(s[:,4], label='MOBIL')
axx.plot(s[:,5], label='IBM')
axx.plot(s[:,6], label='WEYER')
axx.plot(s[:,7], label='CITCRP')
axx.legend(loc='best')
# Get std dev for marginals
stddev_mar = np.zeros(s.shape)
for ii, ss in enumerate(S):
    stddev_mar[ii,:] = np.sqrt(np.diag(ss))


V = np.diag(1./np.diag(np.linalg.inv(V)))
W[nbstock:,nbstock:] = np.diag(1./np.diag(np.linalg.inv(W[nbstock:,nbstock:])))

sample_size = 1000
AllThetas = []
for ii in range(nbstock):
    AllThetas.append(np.zeros((timesteps+1, sample_size)))
Uv, Sv, UvT = np.linalg.svd(V)
svd_invV = [Uv, 1/Sv]
Uw, Sw, UwT = np.linalg.svd(W)
svd_invW = [Uw, 1/Sw]
Uc0, Sc0, Uc0T = np.linalg.svd(C0)
svd_C0 = [Uc0, Sc0]
for ii in range(sample_size):
    thetas, tmp = FFBS(Y, m0, svd_C0, Ft, Gt, svd_invV, svd_invW)
    for jj, AT in enumerate(AllThetas):
        AT[:,ii] = thetas[:,jj+4]
    

fig = plt.figure()
ax = fig.add_subplot(221)
mean = AllThetas[0].mean(axis=1)
std = AllThetas[0].std(axis=1)
ax.plot(mean, 'b')
ax.plot(mean+2*std, 'b--')
ax.plot(mean-2*std, 'b--')
ax.plot(s[:,4], 'k')
ax.plot(s[:,4]+2*stddev_mar[:,4], 'k--')
ax.plot(s[:,4]-2*stddev_mar[:,4], 'k--')
ax.set_title('MOBIL')

ax2 = fig.add_subplot(222)
mean = AllThetas[1].mean(axis=1)
std = AllThetas[1].std(axis=1)
ax2.plot(mean, 'g')
ax2.plot(mean+2*std, 'g--')
ax2.plot(mean-2*std, 'g--')
ax2.plot(s[:,5], 'k')
ax2.plot(s[:,5]+2*stddev_mar[:,5], 'k--')
ax2.plot(s[:,5]-2*stddev_mar[:,5], 'k--')
ax2.set_title('IBM')

ax3 = fig.add_subplot(223)
mean = AllThetas[2].mean(axis=1)
std = AllThetas[2].std(axis=1)
ax3.plot(mean, 'r')
ax3.plot(mean+2*std, 'r--')
ax3.plot(mean-2*std, 'r--')
ax3.plot(s[:,6], 'k')
ax3.plot(s[:,6]+2*stddev_mar[:,6], 'k--')
ax3.plot(s[:,6]-2*stddev_mar[:,6], 'k--')
ax3.set_title('WEYER')

ax4 = fig.add_subplot(224)
mean = AllThetas[3].mean(axis=1)
std = AllThetas[3].std(axis=1)
ax4.plot(mean, 'c')
ax4.plot(mean+2*std, 'c--')
ax4.plot(mean-2*std, 'c--')
ax4.plot(s[:,7], 'k')
ax4.plot(s[:,7]+2*stddev_mar[:,7], 'k--')
ax4.plot(s[:,7]-2*stddev_mar[:,7], 'k--')
ax4.set_title('CITCRP')

mysamples=[100,200,300,400,500]
fig2 = plt.figure()
ax = fig2.add_subplot(221)
ax.plot(AllThetas[0][:,mysamples])
ax.plot(s[:,4], 'k--')
ax.set_title('MOBIL')

ax2 = fig2.add_subplot(222)
ax2.plot(AllThetas[1][:,mysamples])
ax2.plot(s[:,5], 'k--')
ax2.set_title('IBM')

ax3 = fig2.add_subplot(223)
ax3.plot(AllThetas[2][:,mysamples])
ax3.plot(s[:,6], 'k--')
ax3.set_title('WEYER')

ax4 = fig2.add_subplot(224)
ax4.plot(AllThetas[3][:,mysamples])
ax4.plot(s[:,7], 'k--')
ax4.set_title('CITCRP')

plt.show()

