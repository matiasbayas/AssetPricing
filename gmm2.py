# GMM Estimation of Discount Factor Model with Two Moment Conditions
# Matias Bayas-Erazo

# 1. CRRA Utility: u(c) = c**(1-gamma)/(1-gamma)
# Moment conditions come from the EE of agent's problem.
# Assume that observations are iid

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = loadmat('data.mat')
data = data['data']

# Data columns are: [year, consumption growth, market ER, SMB, HML, rf]
cg = data[:,1]
mrkt = data[:,2]
SMB = data[:,3]

# Time periods:
T = mrkt.shape[0]

# Define moment conditions to use in estimation:
def smoments(gamma, uc, Ra, Rb, N):
    # Find sdf:
    m= uc**(-gamma)
    # Now compute E (mR)
    f1 = 1/N *( m[np.newaxis, :]@Ra[:, np.newaxis])
    f2 = 1/N *( m[np.newaxis, :]@Rb[:, np.newaxis])
    aux = np.concatenate([f1,f2])
    F = aux.T@np.eye(2)@aux
    return F

gammas = minimize(lambda x: smoments(x, cg, mrkt, SMB, T), x0 = 10, method = 'Nelder-Mead')

x = np.linspace(35,70,150)
F = np.empty_like(x)
for i in range(150):
    F[i] = smoments(x[i], cg ,mrkt, SMB, T)

plt.plot(x, F*100)
plt.plot(gammas.x, smoments(gammas.x,cg, mrkt, SMB, T)*100, 'ro')
plt.axvline(x= gammas.x, linewidth = 0.7, linestyle = '--', color = 'k')
plt.legend(['Objective: $Q(\gamma)$', 'GMM Estimate'])
plt.xlabel('Risk Aversion - $\gamma$')
plt.ylabel('GMM Objective Function - $Q(\gamma)$')
plt.title('GMM Estimation of Risk Aversion Parameter')
plt.show()

# Find standard error of estimates:
# a. Compute residuals:
m = cg**(-gammas.x)
eps1 = m*mrkt
eps2 = m*SMB
s = np.stack([eps1,eps2])
S = 1/T*(s@s.T)
print(S)

# b. Find gradient:
d = np.zeros([2,1])
d[0,0] = -(1/T)*( (np.log(cg)* m)[np.newaxis,:]@ mrkt[:, np.newaxis] )
d[1,0] = -(1/T)*( (np.log(cg)* m)[np.newaxis,:]@ SMB[:, np.newaxis] )

# c. Compute variance covariance matrix:
V = np.linalg.inv(d.T@np.linalg.inv(S)@d)
print(np.sqrt(V/T))
