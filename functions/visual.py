import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

def correlogram(matrix=None, dec=1, title='Correlogram', valMin=-1, valMax=1):
    plt.figure(title, figsize=(10, 7))
    plt.figure(title, fontsize=8, color='k', verticalalignemt='bottom')
    print(np.round(matrix, dec))
    sb.heatmap(data=np.round(matrix, dec), cmap='Blues',vmin=valMin, vmax=valMax, annot=True)


def principalComponents(eigenvalues=None, columns=None, title='Explained variance of the principal components'):
    plt.figure(title, figsize=(10, 7))
    plt.title(title, fontsize=12, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel='Principal components', fontsize=12, color='k', verticalalignment='top')
    plt.ylabel(ylabel='Eigevalues (variance)', fontsize=12, color='k', verticalalignment='bottom')
    if columns==None:
        components = ['C'+str(i+1) for i in range(len(eigenvalues))]
    plt.plot(components, eigenvalues, 'bo-')
    plt.axhline(y=1, color='r')


def corrCircle(matrix=None, V1=0, V2=1, dec=2,
               labelX=None, labelY=None, valMin=-1, valMax=1, title='Correlation circle'):
    plt.figure(title, figsize=(7, 7))
    plt.title(title, fontsize=10, color='k', verticalalignment='bottom')
    T = [t for t in np.arange(0, np.pi*2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(y=0, color='g')
    plt.axvline(x=0, color='g')
    if labelX==None or labelY==None:
        if isinstance(matrix, pd.DataFrame):
            plt.xlabel(xlabel=matrix.columns[V1], fontsize=12, color='k', verticalalignment='top')
            plt.ylabel(ylabel=matrix.columns[V2], fontsize=12, color='k', verticalalignment='bottom')
        else:
            plt.xlabel(xlabel='Var '+str(V1+1), fontsize=12, color='k', verticalalignment='top')
            plt.ylabel(ylabel='Var '+str(V2+1), fontsize=12, color='k', verticalalignment='bottom')
    else:
        plt.xlabel(xlabel=labelX, fontsize=12, color='k', verticalalignment='top')
        plt.ylabel(ylabel=labelY, fontsize=12, color='k', verticalalignment='bottom')

    if isinstance(matrix, np.ndarray):
        plt.scatter(x=matrix[:, V1], y=matrix[:, V2], c='r', vmin=valMin, vmax=valMax)
        for i in range(matrix.shape[0]):
            plt.text(x=matrix[i, V1], y=matrix[i, V2],
                     s='('+str(round(matrix[i, V1], dec)) +
                                                         ', ' + str(round(matrix[i, V2], dec)) + ')')
    if isinstance(matrix, pd.DataFrame):
        plt.scatter(x=matrix.iloc[:, V1], y=matrix.iloc[:, V2],
                    c='r', vmin=valMin, vmax=valMax)
        for i in range(matrix.values.shape[0]):
            plt.text(x=matrix.iloc[i, V1], y=matrix.iloc[i, V2], s=matrix.index[i])


def show():
    plt.show()