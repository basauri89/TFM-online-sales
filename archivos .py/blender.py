"""
Simple blender para los valores de regresion deseados durante meses

"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import load_data
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
import pickle
import gzip
import math

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

if __name__ == '__main__':
    
    #iniciamos la seed para la aleatoriedad y creamos un 5 fold cross validation

    np.random.seed(0)
    n_folds = 5
    
    #cagamos el dataset

    X, X_submission, ys, ids, idx = load_data.load()    
    
    # evitamos el logscale en la evaluacion:
    ys = np.log(ys/500.0 + 1.0)      
    y_submission = np.zeros((X_submission.shape[0], 12))    

      #se prueba con n stimators 1000 para que se ejecute más rápido
    regs = [GradientBoostingRegressor(learning_rate=0.001, subsample=0.5, max_depth=6, n_estimators=10000)]

    dataset_blend_train = np.zeros((X.shape[0], 12*len(regs)), dtype=np.double)
    dataset_blend_submission = np.zeros((X_submission.shape[0], 12*len(regs), n_folds), dtype=np.double)
    
    
    for i in range(12):
        print "Month", i
        y = ys[:,i]
        kfcv = KFold(n=X.shape[0], n_folds=n_folds)
        for j, (train, test) in enumerate(kfcv):
            print "Fold", j
            for k, reg in enumerate(regs):
                print reg
                #Nos aseguramos de eliminar todos los valores infinitos o NaN
                y[train] = np.nan_to_num(y[train])
                X[train] = np.nan_to_num(X[train])
                X[test] = np.nan_to_num(X[test])
                X_submission = np.nan_to_num(X_submission)
                #check de valores NaN o infinitos
                print "y tiene valores infinitos: ", np.isinf(y[train]).any()
                print "y tiene valores nan: ", np.isnan(y[train]).any()
                print "X tiene valores nan: ", np.isnan(X[train]).any()
                print "X tiene valores infinitos: ", np.isnan(X[train]).any()                
                reg.fit(X[train], y[train])
                #ejecutamos el predictor
                dataset_blend_train[test,12*k+i] = reg.predict(X[test])
                dataset_blend_submission[:,12*k+i,j] = reg.predict(X_submission)

    
    dataset_blend_submission_final = dataset_blend_submission.mean(2)
    print "dataset_blend_submission_final:", dataset_blend_submission_final.shape

    print "Blending."
    for i in range(12):
        print "Month", i, '-',
        y = ys[:,i]
        reg = RidgeCV(alphas=np.logspace(-2,4,40))
        reg.fit(dataset_blend_train, y)
        print "best_alpha =", reg.alpha_
        y_submission[:,i] = reg.predict(dataset_blend_submission_final)
                
    # reconversion de los resultados a la dimension original:
    y_submission = (np.exp(y_submission) - 1.0) * 500.0
    
    print "Guardando resultados en test.csv..."
    np.savetxt("test.csv", np.hstack([ids[:,None], y_submission]), fmt="%d", delimiter=',')
    print("Resultados guardados en test.csv")
    yreal = (np.exp(dataset_blend_submission_final) - 1.0) * 500.0
    print rmsle(yreal, y_submission)
    
    
    
    
