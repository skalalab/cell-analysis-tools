from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import umap
import warnings
from pathlib import Path
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd


import seaborn as sns

import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
    
#%%

if __name__ == "__main__":
    

    #%% SELECT/GENERATE DATA
    x, y = make_classification(n_samples=1000, n_features=10,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)

    list_features = [
        # 'nadh_intensity_mean',
        'nadh_a1_mean',  
        'nadh_a2_mean',
        'nadh_t1_mean',  
        'nadh_t2_mean',
        'nadh_tau_mean_mean', 
        # 'fad_intensity_mean',  
        'fad_a1_mean',
        'fad_a2_mean',  
        'fad_t1_mean',
        'fad_t2_mean',  
        'fad_tau_mean_mean',
        # 'redox_ratio_mean'
        ]
    df_data = pd.DataFrame(x, columns=list_features)
    
    #%% Scale data 
    df_data_scaled = pd.DataFrame(StandardScaler().fit_transform(df_data),columns=list_features) 
    
    #%% Split into test/train
    x_train, x_test, y_train, y_test = train_test_split(df_data_scaled.values, y,
                     train_size=0.8
                     )
    
    #%% RANDOM FOREST 
    
    ##%% MAKE DICT OF PARAMETERS TO EXPLORE
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    
    paramter_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    ##%% RUN CLASSIFIER
    
    # Random Forest Classification 
    print('Random Forest Classification') 
    
    #Initializes a linear support vector machine classifier      
    rfc = RandomForestClassifier()
    
    # NON-Exhaustively searches over specific parameter range  for  best estimator for classification on training data with 3-fold cross validation      
    rf_random = RandomizedSearchCV(estimator=rfc, param_distributions = paramter_grid, n_iter = 100, cv = 3, random_state=42)
    rf_random.fit(x_train, y_train)
    
    print(f"Best parameters set found on development set: {rf_random.best_params_}")
    print(f"Accuracy on the test set with raw data: {rf_random.score(x_test, y_test):.5f}")
    
    #Determines likelihood of belonging to each class       
    rf_raw_decision_scores = rf_random.predict_proba(x_test)
    
    #Determines false positive rate and true positive rate for plotting ROC curves
    fprRF, tprRF, thresRF = roc_curve(y_test, rf_raw_decision_scores[:,1])

    # plot feature importances
    array_importances = rf_random.best_estimator_.feature_importances_
    for feature, importance in zip(list_features, array_importances):
        pass
        plt.bar(feature, importance)
        plt.xticks(rotation=90)
        plt.xlabel("features")
        plt.ylabel("percent")
    plt.title("Feature Importance")
    plt.show()
    
    #%% SVM
    print('SVM Classification')
    
    #Initializes a linear support vector machine classifier
    svc = SVC(kernel = "linear", probability = True)
    
    #Provides the bounds of classifier parameters to iterate through to optimize classifier performance
    param_grid = {'C': [10 ** k for k in range(-3, 4)]}
    
    
    #Exhaustively searches over specific parameter range  for  best estimator for classification on training data with default 5-fold cross validation
    clfSV = GridSearchCV(svc, param_grid)
    clfSV.fit(x_train, y_train)
    
    print("Best parameters set found on development set:")
    print(clfSV.best_params_)
          
    print(f"Accuracy on the test set with raw data: {clfSV.score(x_test, y_test):.5f}")
    
    #Determines decision function from classification (how far away each point is from hyperplane separating classes and to which side of hyperplane [class] the point belongs)       
    SVraw_decision_scores = clfSV.decision_function(x_test)
    #Determines false positive rate and true positive rate for plotting ROC curves
    fprSV, tprSV, thresSV = roc_curve(y_test, SVraw_decision_scores, pos_label = 1)
    
    # plot feature importances
    array_importances = clfSV.best_estimator_.coef_.squeeze()
    for feature, importance in zip(list_features, array_importances):
        pass
        plt.bar(feature, abs(importance))
        plt.xticks(rotation=90)
        plt.xlabel("features")
        plt.ylabel("coefficient")
    plt.title("Feature Importance | SVM")
    plt.show()

    #%% Logistic Regression Classification 
    print("Logistic Regression Classification")
    
    #Initializes a logistic regression classifier  
    lrc = LogisticRegression()
    
    #Provides the bounds of classifier parameters to iterate through to optimize classifier performance
    param_lrc = {'penalty': ['l2'],
                  'C': np.logspace(-4,4,20),
                  'solver':['newton-cg', 'lbfgs','sag','saga']}
    
    #Exhaustively searches over specific parameter range  for  best estimator for classification on training data with default 5-fold cross validation      
    clfLR = GridSearchCV(estimator = LogisticRegression(), param_grid = param_lrc, cv = 3)
    clfLR.fit(x_train, y_train)
          
          
    print(clfLR.best_params_)
    print(f"Accuracy on the test set with raw data: {clfLR.score(x_test, y_test):.5f}")
    
    #Determines decision function from classification (how far away each point is from hyperplane separating classes and to which side of hyperplane [class] the point belongs)      
    lr_raw_decision_scores = clfLR.decision_function(x_test)
    #Determines false positive rate and true positive rate for plotting ROC curves
    fprLR, tprLR, thresLR = roc_curve(y_test, lr_raw_decision_scores, pos_label = 1)

    # plot feature importances
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    array_importances = clfLR.best_estimator_.coef_.squeeze()
    for feature, importance in zip(list_features, array_importances):
        pass
        plt.bar(feature, abs(importance))
        plt.xticks(rotation=90)
        plt.xlabel("features")
        plt.ylabel("coefficient")
    plt.title("Feature Importance | Linear Regression")
    plt.show()


    # ROC curves 
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
#%% SVM


    ## Support Vector Machine Classification 
    print('SVM Classification')
    
    
    #Initializes a linear support vector machine classifier
    svc = SVC(kernel = "linear", probability = True)
    
    #Provides the bounds of classifier parameters to iterate through to optimize classifier performance
    params_grid = {'C': [10 ** k for k in range(-3, 4)]}
    
    
    #Exhaustively searches over specific parameter range  for  best estimator for classification on training data with default 5-fold cross validation
    clfSV = GridSearchCV(svc, params_grid)
    clfSV.fit(x_train, y_train)
    
    print("Best parameters set found on development set:")
    print(clfSV.best_params_)
          
          
    print(f"Accuracy on the test set with raw data: {clfSV.score(x_test, y_test):.5f}")
    
    #Determines decision function from classification (how far away each point is from hyperplane separating classes and to which side of hyperplane [class] the point belongs)       
    SVraw_decision_scores = clfSV.decision_function(x_test)
    #Determines false positive rate and true positive rate for plotting ROC curves
    fprSV, tprSV, thresSV = roc_curve(y_test, SVraw_decision_scores, pos_label = 1)
    
    #%% LOGISTIC REGRESSION
    
        #Logistic Regression Classification 
    print("Logistic Regression Classification")
    
    #Initializes a logistic regression classifier  
    lrc = LogisticRegression()
    
    #Provides the bounds of classifier parameters to iterate through to optimize classifier performance
    param_lrc = {'penalty': ['l2'],
                  'C': np.logspace(-4,4,20),
                  'solver':['newton-cg', 'lbfgs','sag','saga']}
    
          
    #Exhaustively searches over specific parameter range  for  best estimator for classification on training data with default 5-fold cross validation      
    clfLR = GridSearchCV(estimator = LogisticRegression(), param_grid = param_lrc, cv = 3)
    clfLR.fit(x_train, y_train)
          
          
    print(clfLR.best_params_)
    print(f"Accuracy on the test set with raw data: {clfLR.score(x_test, y_test):.5f}")
    
    #Determines decision function from classification (how far away each point is from hyperplane separating classes and to which side of hyperplane [class] the point belongs)      
    lr_raw_decision_scores = clfLR.decision_function(x_test)
    #Determines false positive rate and true positive rate for plotting ROC curves
    fprLR, tprLR, thresLR = roc_curve(y_test, lr_raw_decision_scores, pos_label = 1)
    
    #%% PLOT ALL ESTIMATORS 
        
    #Generates roc curve to compare performance of all 3 classifiers based on their optimized parameters
    plt.plot(fprSV, tprSV, "r", label=f'Linear SVM, AUC:{roc_auc_score(y_test, SVraw_decision_scores):.5f}')
    plt.plot(fprRF, tprRF, "g", label=f'Random Forest, AUC:{roc_auc_score(y_test, rf_raw_decision_scores[:,1]):.5f}')
    plt.plot(fprLR, tprLR, "y", label=f'Logistic Regression, AUC:{roc_auc_score(y_test, lr_raw_decision_scores):.5f}')
    plt.plot([0,1],[0,1], "k--", label='Random Guess')
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")

    # plt.savefig('ROC curve by day_ALL.png') #saves tif of ROC curves
  
