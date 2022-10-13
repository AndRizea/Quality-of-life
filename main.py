import pandas as pd
from functions import replaceNaN as r
from PCA import pca
from functions import visual as vi

data = pd.read_excel('dataIN.xlsx', index_col=0)

r.replaceNAN(data)
data.to_excel('Output/dataOUT.xlsx')
print(data)

variables = list(data.columns)
countries = list(data.index)

matrix_X = data[variables].values
#print(matrix_X)

pca_inst = pca.PCA(matrix_X)

#matrix X standardized
matrix_Xstd = pca_inst.getXstd()
matrix_Xstd_df = pd.DataFrame(data=matrix_Xstd, index=countries, columns=variables)
matrix_Xstd_df.to_csv('Output/matrix_Xstd.csv')

#covariance matrix X
matrix_Xcov = pca_inst.getCov()
matrix_Xcov_df = pd.DataFrame(data=matrix_Xcov, index=variables, columns=variables)
matrix_Xcov_df.to_csv('Output/matrix_Xcov.csv')
vi.correlogram(matrix=matrix_Xcov_df, dec=2, title='Covariance matrix')
vi.show()

principal_components = pca_inst.getPrincipalComp()
eigenvalues = pca_inst.getEigenvalues()

#graphic of the explained variance by the principal components
vi.principalComponents(eigenvalues=eigenvalues)
vi.show()

#correlogram of factor loadings
Rxc = pca_inst.getFactorLoadings()
Rxc_df = pd.DataFrame(data=Rxc, index=variables, columns=('C' + str(i+1) for i in range(Rxc.shape[1])))
Rxc_df.to_csv('Output/FactorLoadings.csv')
vi.correlogram(matrix=Rxc_df, dec=2, title='Correlogram of factor loadings')
vi.show()

#correlogram for the quality of points representation on axis
pointsQual = pca_inst.getPointsQual()
pointsQual_df = pd.DataFrame(data=pointsQual, index=countries, columns=('C' + str(i+1) for i in range(Rxc.shape[1])))
vi.correlogram(matrix=pointsQual_df, dec=2, title='Correlogram for the quality of points representation on axis of principal components')
vi.show()

#correlogram of betas
beta = pca_inst.getBeta()
beta_df = pd.DataFrame(data=beta, index=countries, columns=('C' + str(i+1) for i in range(Rxc.shape[1])))
vi.correlogram(matrix=beta_df, dec=2, title='Correlogram of betas')
vi.show()

#correlogram of commonalities
commonalities = pca_inst.getCommon()
commonalities_df = pd.DataFrame(data=commonalities, index=variables, columns=('C' + str(i+1) for i in range(Rxc.shape[1])))
vi.correlogram(matrix=commonalities_df, dec=2, title='Correlogram of commonalities')
vi.show()

#correlation circle for the initial variables in the space of C1 and C2
vi.corrCircle(matrix=Rxc_df, title='Correlation circle for the initial variables in the space of C1 and C2')
vi.show()