# GMM Estimation of Simple SDF Model with One Moment Condition
# Matias Bayas-Erazo

# 1. CRRA Utility: u(c) = c**(1-gamma)/(1-gamma)
# Moment conditions come from the EE of agent's problem.
# Assume that observations are iid

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize

data = loadmat('data.mat')
data = data['data']

# Data columns are: year, consumption growth, market ER, SMB, HML, rf
cg = data[:,1]
mrkt = data[:,2]

# Time periods:
T = mrkt.shape[0]

# Define moment condition to use in estimation:
def smoments(gamma, uc, R, N):
    # Find sdf:
    m= uc**(-gamma)
    # Now compute E (mR)
    f = 1/N *( m[np.newaxis, :]@R[:, np.newaxis])

    return f

# The function is decreasing in gamma:
# Compute function to plot below
x = np.linspace(10,70,150)
F = np.empty_like(x)
for i in range(150):
    F[i] = smoments(x[i], cg ,mrkt, T)


# Finding the gamma that sets EE = 0:
gammas = scipy.optimize.broyden1(lambda x: smoments(x, cg, mrkt, T), 10, f_tol = 1e-9)

plt.plot(x, F)
plt.plot(gammas, smoments(gammas,cg, mrkt, T), 'ro')
plt.legend(['FOC', 'GMM Estimate'])
plt.xlabel('Risk Aversion - $\gamma$')
plt.ylabel('SDF Adjusted Returns - $E(m_{t+1} R_{t+1})$')
plt.title('GMM Estimation of Risk Aversion Parameter')
plt.axvline(x= gammas, linewidth = 0.7, linestyle = '--', color = 'k')
plt.show()
