import numpy as np

class PCA:

    def __init__(self, X):
        self.X = X

        #standardize the values
        avg = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        self.Xstd = (self.X - avg) / std

        #compute covariance
        self.Cov = np.cov(self.Xstd, rowvar=False)

        #compute the eigenvalues and the eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.Cov)

        #sort the eigenvalues descedingly
        eigenv_des = [i for i in reversed(np.argsort(eigenvalues))]
        self.alpha = eigenvalues[eigenv_des]
        self.a = eigenvectors[:, eigenv_des]

        #regularization of the eigenvectors
        for col in range(self.a.shape[1]):
            minimum = np.min(self.a[:, col])
            maximum = np.max(self.a[:, col])

            if (np.abs(minimum) > np.abs(maximum)):
                self.a[:, col] = -self.a[:, col]


        #compute the principal components
        self.C = self.Xstd @ self.a

        #FACTOR LOADINGS
        #scores
        self.Cstd = self.C / np.sqrt(self.alpha)

        #correlation between the initial variables and the principle components
        self.Rxc = self.a * np.sqrt(self.alpha)

        #quality of points representation on the axis of the principal components
        C2 = self.C * self.C
        C2sum = np.sum(C2, axis=1)
        self.PointsQual = np.transpose(np.transpose(C2) / C2sum)

        #contribution of observations to the variance of axis
        self.beta = C2 / (self.X.shape[0] * self.alpha)

        #commonalities
        Rxc2 = self.Rxc * self.Rxc
        self.common = np.cumsum(Rxc2, axis=1)

    def getXstd(self):
        return self.Xstd

    def getCov(self):
        return self.Cov

    def getEigenvalues(self):
        return self.alpha

    def getEigenvectors(self):
        return self.a

    def getPrincipalComp(self):
        return self.C

    def getScores(self):
        return self.Cstd

    def getPointsQual(self):
        return self.PointsQual

    def getBeta(self):
        return self.beta

    def getFactorLoadings(self):
        return self.Rxc

    def getCommon(self):
        return self.common


