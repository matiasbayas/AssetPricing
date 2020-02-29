import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy.io import loadmat

factors = loadmat('FF_factors.mat')
portfolios = loadmat('portfolios.mat')

factors = factors['FF_factors']
portfolios = portfolios['portfolios']

# Get market excess return, SMB, HML, and Rf from the factors data:
mrkt = factors[:,0]
smb = factors[:,1]
hml = factors[:,2]
rf = factors[:,3]

# Find dimensions of sample:
T = mrkt.shape[0]
N = portfolios.shape[1]

# Excess return matrix:
Re = np.empty_like(portfolios)
for i in range(N):
    Re[:,i] = portfolios[:,i] - rf

# Arrays to store betas for each factor and pricing errors (alpha):
betas = np.zeros([25,3])
alphas = np.zeros([25,1])

# Add constant to matrix:
factors2 = sm.add_constant(factors)

# Run time series regression for each asset:
for i in range(N):
    model = sm.OLS(endog = portfolios[:,i], exog = factors2)
    results = model.fit()
    alphas[i,:] = results.params[0]
    betas[i,:] = results.params[1:4]

print(results.summary())

# Find the price of risk, using estimated betas:

# 1 - NAIVE Estimation: (doesn't take into account betas are estimated)
avg_ret = Re.mean(axis = 0)
model = sm.OLS(endog = avg_ret, exog = betas)
results = model.fit()
print(results.summary())

yer = results.conf_int(alpha = 0.05)

plt.errorbar(['Market', 'SMB', 'HML'],results.params,
                yerr = [yer[0,0] - results.params[0],yer[1,0] - results.params[1],
                        yer[2,0] - results.params[2]], fmt = 'bo', ecolor = 'r',
                        capsize = 4.5, marker = 'd')

plt.xticks(np.arange(3), ['Market', 'SMB', 'HML'])
plt.ylim([0,0.7])
plt.title('Estimated Price of Risk in Fama-French 3-Factor Model')
plt.show()
