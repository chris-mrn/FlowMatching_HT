import random
from scipy.stats import norm, t
import scipy.stats as stats
import numpy as np
import openturns as ot
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.correlation_tools import cov_nearest

#1. all dimensions are independent of each other
class independent_generator:
    def __init__(self, D):
        self.D = D

    def get_data(self, n):
        self.mv_samps = np.zeros((n, self.D))
        for j in range(self.D):
            if j<int(self.D/4):
                mean = np.random.rand() * 8 - 4 # sample random mean between -4 and 4
                sd = np.random.rand() # sample sd from 0 to 8
                samp = norm(mean, sd).rvs(size=n)
            elif j<int(3*self.D/8):
                mean1 = np.random.rand() * 8 - 4
                mean2 = np.random.rand() * 8 - 4
                sd1 = np.random.rand()
                sd2 = np.random.rand()
                samp = np.concatenate([norm(mean1, sd1).rvs(size=int(n/2)), norm(mean2, sd2).rvs(size=int(n/2))])
                np.random.shuffle(samp)
            elif j<int(self.D/2):
                mean1 = np.random.rand() * 8 - 4
                mean2 = np.random.rand() * 8 - 4
                mean3 = np.random.rand() * 8 - 4
                sd1 = np.random.rand()
                sd2 = np.random.rand()
                sd3 = np.random.rand()
                samp = np.concatenate([norm(mean1, sd1).rvs(size=int(n / 3)), norm(mean2, sd2).rvs(size=int(n / 3)), norm(mean3, sd3).rvs(size=int(n / 3))])
                np.random.shuffle(samp)
            elif j<int(3*self.D/4):
                mean = np.random.rand() * 8 - 4
                sd = np.random.rand()
                samp = t(df=4, loc=mean, scale=sd).rvs(size=n)
            else:
                mean1 = np.random.rand() * 8 - 4
                mean2 = np.random.rand() * 8 - 4
                sd1 = np.random.rand()
                sd2 = np.random.rand()
                samp = np.concatenate([t(df=4, loc=mean1, scale=sd1).rvs(size=int(n / 2)), norm(mean2, sd2).rvs(size=int(n / 2))])
                np.random.shuffle(samp)
            self.mv_samps[:, j] = samp

        # scale the data:
        scaler = StandardScaler()
        self.mv_samps = scaler.fit_transform(self.mv_samps)
        return self.mv_samps

    def visualize_marginals(self, save=False):
        # plot the marginals:
        fig, axs = plt.subplots(2, int(self.D/2))
        fig.suptitle("Independent Components")
        j = 0
        for ax in axs.flat:
            ax.hist(self.mv_samps[:, j], range=(-5, 5), bins=30, density=True)
            j += 1
        if save:
            plt.savefig("plots/marginals")
        else:
            plt.show()

'''
data_generator = independent_generator(D, n)
data = data_generator.get_data()
data_generator.visualize_marginals()
'''

class copula_generator:
    def __init__(self, D, num_heavy, df_t=2, seed=1):
        self.D = D
        self.marginals = []

        # correlation matrix with random 0.25 on off-diag
        np.random.seed(seed)
        self.R = ot.CorrelationMatrix(D)
        if self.D!=50:
            for i in range(2*D): # number of non-zero entries = 2*D
                j = np.random.randint(0, self.D)
                k = np.random.randint(0, self.D)
                if j != k:
                    self.R[j, k] = 0.25
                    self.R[k, j] = 0.25
        else: # i.e. self.D==50
            # randomly select 30 entries in the lower right 10x10 correlation matrix block
            if num_heavy==5:
                for i in range(30):
                    j = np.random.randint(0, 10) + 40
                    k = np.random.randint(0, 10) + 40
                    if j != k:
                        self.R[j, k] = 0.25
                        self.R[k, j] = 0.25
            elif num_heavy==10:
                R = np.diag([1.0]*self.D)
                for i in range(4*D):
                    j = np.random.randint(0, 20) + 30
                    k = np.random.randint(0, 20) + 30
                    if j != k:
                        R[j, k] = 0.25
                        R[k, j] = 0.25

                # ensure that R is a positive define matrix:
                R = cov_nearest(R)
                self.R = ot.CorrelationMatrix(self.D, R.flatten())
        self.copula = ot.NormalCopula(self.R)

        self.df_t = df_t
        # define the multivariate distribution
        if self.D!=50:
            for j in range(self.D):
                if j < int(self.D / 4):
                    mean = np.random.rand() * 8 - 4  # sample random mean between -4 and 4
                    sd = np.random.rand() + 1  # sample sd from 0.2 to 1.2
                    self.marginals.append(ot.Normal(mean, sd))
                elif j < int(3 * self.D / 8):
                    mean1 = np.random.rand() * 8 - 4
                    mean2 = np.random.rand() * 8 - 4
                    sd1 = np.random.rand() + 1
                    sd2 = np.random.rand() + 1
                    weights = [0.5, 0.5]
                    mixture_comps = [ot.Normal(mean1, sd1), ot.Normal(mean2, sd2)]
                    self.marginals.append(ot.Mixture(mixture_comps, weights))
                elif j < int(self.D / 2):
                    mean1 = np.random.rand() * 8 - 4
                    mean2 = np.random.rand() * 8 - 4
                    mean3 = np.random.rand() * 8 - 4
                    sd1 = np.random.rand() + 1
                    sd2 = np.random.rand() + 1
                    sd3 = np.random.rand() + 1
                    weights = [1 / 3, 1 / 3, 1 / 3]
                    mixture_comps = [ot.Normal(mean1, sd1), ot.Normal(mean2, sd2), ot.Normal(mean3, sd3)]
                    self.marginals.append(ot.Mixture(mixture_comps, weights))

                # new:
                elif j>=self.D - num_heavy:
                    mean1 = np.random.rand() * 8 - 4
                    mean2 = np.random.rand() * 8 - 4
                    sd1 = np.random.rand() + 1
                    sd2 = np.random.rand() + 1
                    weights = [0.5, 0.5]
                    mixture_comps = [ot.Student(df_t, mean1, sd1), ot.Student(df_t, mean2, sd2)]
                    self.marginals.append(ot.Mixture(mixture_comps, weights))
                else:
                    mean1 = np.random.rand() * 8 - 4
                    mean2 = np.random.rand() * 8 - 4
                    sd1 = np.random.rand() + 1
                    sd2 = np.random.rand() + 1
                    weights = [0.5, 0.5]
                    #mixture_comps = [ot.Student(df_t, mean1, sd1), ot.Normal(mean2, sd2)]
                    # new:
                    mixture_comps = [ot.Normal(mean1, sd1), ot.Normal(mean2, sd2)]
                    self.marginals.append(ot.Mixture(mixture_comps, weights))
        else:# i.e. self.D==50
            for j in range(self.D):
                # first 40 components are just gaussians,
                # next 5 are mixture of 2 gaussians,
                # last 5 are mixture of 2 student t
                if num_heavy==5:
                    if j < 40:
                        mean = np.random.rand() * 8 - 4  # sample random mean between -4 and 4
                        sd = np.random.rand() + 1  # sample sd from 0.2 to 1.2
                        self.marginals.append(ot.Normal(mean, sd))
                    elif j < 45:
                        mean1 = np.random.rand() * 8 - 4
                        mean2 = np.random.rand() * 8 - 4
                        sd1 = np.random.rand() + 1
                        sd2 = np.random.rand() + 1
                        weights = [0.5, 0.5]
                        mixture_comps = [ot.Normal(mean1, sd1), ot.Normal(mean2, sd2)]
                        self.marginals.append(ot.Mixture(mixture_comps, weights))
                    else:
                        mean1 = np.random.rand() * 8 - 4
                        mean2 = np.random.rand() * 8 - 4
                        sd1 = np.random.rand() + 1
                        sd2 = np.random.rand() + 1
                        weights = [0.5, 0.5]
                        mixture_comps = [ot.Student(df_t, mean1, sd1), ot.Student(df_t, mean2, sd2)]
                        self.marginals.append(ot.Mixture(mixture_comps, weights))
                elif num_heavy==10:
                    if j < 40:
                        mean1 = np.random.rand() * 8 - 4
                        mean2 = np.random.rand() * 8 - 4
                        sd1 = np.random.rand() + 1
                        sd2 = np.random.rand() + 1
                        weights = [0.5, 0.5]
                        mixture_comps = [ot.Normal(mean1, sd1), ot.Normal(mean2, sd2)]
                        self.marginals.append(ot.Mixture(mixture_comps, weights))
                    else:
                        mean1 = np.random.rand() * 8 - 4
                        mean2 = np.random.rand() * 8 - 4
                        sd1 = np.random.rand() + 1
                        sd2 = np.random.rand() + 1
                        weights = [0.5, 0.5]
                        mixture_comps = [ot.Student(df_t, mean1, sd1), ot.Student(df_t, mean2, sd2)]
                        self.marginals.append(ot.Mixture(mixture_comps, weights))


        self.dist = ot.ComposedDistribution(self.marginals, self.copula)
        self.mv_samps = np.array([])

    def get_data(self, n):
        self.mv_samps = np.array(self.dist.getSample(n))
        # a small test...
        # self.mv_samps = np.flip(self.mv_samps)
        return self.mv_samps

    def get_marginals(self):
        return self.marginals

    def visualize_marginals(self, save=False, range_x=5):
        fig, axs = plt.subplots(2, int(self.D / 2))
        fig.suptitle("Dependency induced by Gaussian Copula")
        j = 0
        for ax in axs.flat:
            ax.hist(self.mv_samps[:, j], range=(-range_x, range_x), bins=30, density=True)
            j += 1
        if save:
            plt.savefig("plots/marginals_copula")
        else:
            plt.show()

    def get_R(self):
        return(self.R)

    def get_dist(self):
        return self.dist


if __name__=="__main__":
    data_generator = copula_generator(50, num_heavy=10, df_t=3)
    data = data_generator.get_data(10000)
    data_generator.visualize_marginals(save=False, range_x=10)
    """
    import pandas as pd
    
    for j in range(16):
    for j in range(16):
        marginal = np.abs(data[:, j])
        df = pd.DataFrame()
        df["data"] = marginal
        df["helper"] = np.repeat(1, len(marginal))
        np.savetxt("data/marginal" + str(j + 1) + ".dat", df.values, fmt=["%10.5f", "%d"])
    """
