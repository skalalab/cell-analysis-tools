from sklearn.metrics import roc_curve, auc,  accuracy_score, classification_report # confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import pandas as pd
from copy import deepcopy
import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import seaborn as sns



#%%  ######## CLASSIFIER FUNCTION
def run_analysis_on_classifier(clf, X_test, y_test, dict_classes):
    print(clf)
    y_proba = clf.predict_proba(X_test)
    # y_pred  = clf.predict(X_test) # no control over operating point
    
    # check probab and y_pred 
    
    # bool_proba_pred_agreement = (y_pred == (y_proba[:,1] > 0.5)).all()
    # if bool_proba_pred_agreement:
    #     print("proba same as pred then thresholded at 0.5")
    # else:
    #     print(f"{'*' * 40} proba NOT the same as pred then thresholded at 0.5")
    
    actual_neg = y_test == 0
    actual_pos = y_test == 1
    print('=' * 41)
    print(f"{'Actual':^15} | Total : {len(y_test)} | CD68- : {sum(actual_neg)}  | CD68+ : {sum(actual_pos)}")
    prob_cd69pos = y_proba[:,1] # these are CD69+ probabilities
    

    # Compute ROC curve and AUC for each class
    print(f"y_test : {len(y_test)} | prob_cd69pos len : {len(prob_cd69pos)}")
    fpr, tpr, array_thresh = roc_curve(y_test, prob_cd69pos) # get label of CD69+, 2nd class
    roc_auc = auc(fpr, tpr)

    
    # ########## ECG trying out probability thresholds
    print('-' * 41, "Emmanuel analysis")
    threshold = 0.50
    pred_neg = prob_cd69pos <= threshold
    pred_pos = prob_cd69pos > threshold
        
    
    TPR = sum(pred_pos * actual_pos) / sum(actual_pos)
    FPR = sum(pred_pos * actual_neg) / sum(actual_neg)
    print(f"Threshold {threshold:.2f} |  CD69- : {sum(pred_neg)} | CD69+ {sum(pred_pos)} | TPR: {TPR:.3f} FPR: {FPR:.3f}")
    print('=' * 41)

    print('-' * 41, "python analysis")
    y_test = pd.Series(y_test.squeeze()).map(dict_classes)
    # y_pred = pd.Series(np.array(pred_pos).astype(int)).map(dict_classes)
    y_pred = pd.Series(np.array(pred_pos, dtype=np.int)).map(dict_classes)
    
    confusion_matrix = pd.crosstab(y_test,
                                   y_pred,
                                   rownames=['Actual'], 
                                   colnames=['Predicted']
                                   )
    print(confusion_matrix)
  
    #print metrics to assess classifier performance
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy score =', accuracy)
    print(classification_report(y_test, y_pred))

    return fpr, tpr, roc_auc, accuracy, (FPR, TPR) # (operating_point_fpr, operating_point_tpr)#



# #%%  ######## CLASSIFIER FUNCTION
# def run_analysis_on_classifier(clf, X_test, y_test, dict_classes):
#     print(clf)
#     y_proba = clf.predict_proba(X_test)
#     # y_pred  = clf.predict(X_test) # no control over operating point
    
#     # check probab and y_pred 
    
#     # bool_proba_pred_agreement = (y_pred == (y_proba[:,1] > 0.5)).all()
#     # if bool_proba_pred_agreement:
#     #     print("proba same as pred then thresholded at 0.5")
#     # else:
#     #     print(f"{'*' * 40} proba NOT the same as pred then thresholded at 0.5")
    
#     actual_neg = y_test == 0
#     actual_pos = y_test == 1
#     print('=' * 41)
#     print(f"{'Actual':^15} | Total : {len(y_test)} | CD68- : {sum(actual_neg)}  | CD68+ : {sum(actual_pos)}")
#     prob_cd69pos = y_proba[:,1] # these are CD69+ probabilities
    

#     # Compute ROC curve and AUC for each class
#     print(f"y_test : {len(y_test)} | prob_cd69pos len : {len(prob_cd69pos)}")
#     fpr, tpr, array_thresh = roc_curve(y_test, prob_cd69pos) # get label of CD69+, 2nd class
#     roc_auc = auc(fpr, tpr)
#     # print(f"{'predicted':^15} |  CD69- : {sum(y_pred==0)} | CD69+ {sum(y_pred==1)} ")
#     # print(f" Total predicted : {len(y_pred)}  | fpr count: {len(fpr)} | tpr count: {len(tpr)}")
    
    
#     # ########## ECG trying out probability thresholds
#     print('-' * 41, "Emmanuel analysis")
#     threshold = 0.50
#     pred_neg = prob_cd69pos <= threshold
#     pred_pos = prob_cd69pos > threshold
    
#     # print((pred_pos == y_pred).all()) # check to see if predictions y sklearn same as >0.5 threshold
    
    
#     TPR = sum(pred_pos * actual_pos) / sum(actual_pos)
#     FPR = sum(pred_pos * actual_neg) / sum(actual_neg)
#     print(f"Threshold {threshold:.2f} |  CD69- : {sum(pred_neg)} | CD69+ {sum(pred_pos)} | TPR: {TPR:.3f} FPR: {FPR:.3f}")
#     print('=' * 41)

#     # ############## Kayvan - Cost function
#     # print('-' * 41, "Kayvan analysis")
#     # # idx = np.argwhere(array_thresh ==0.5)
#     # # cost = fpr - tpr # 45 degree line adjacent to roc, top left down
    
#     # # weighted by number of members of each class 
#     # # minimizes off diagonal of confusion matrix, fp, fn misclassifications
#     # cost = fpr * sum(actual_neg) - tpr * sum(actual_pos) 
#     # idx_min = np.argmin(cost)
    
#     # thresh_optimum = array_thresh[idx_min]
#     # operating_point_fpr = fpr[idx_min]
#     # operating_point_tpr = tpr[idx_min]
#     # # print(f"fpr: {operating_point_fpr:.3f} | tpr : {operating_point_tpr:.3f} | operating threshold: {thresh_optimum}")
    
#     # # scores_cd69 = y_proba[:,1] # these are CD69+ probabilities
#     # pred_neg = prob_cd69pos < thresh_optimum
#     # pred_pos = prob_cd69pos >= thresh_optimum
#     # print(f"Threshold {thresh_optimum:.2f} |  CD69- : {sum(pred_neg)} | CD69+ {sum(pred_pos)} | | TPR: {operating_point_tpr:.3f} FPR: {operating_point_fpr:.3f}")
#     # ##############

#     print('-' * 41, "python analysis")
#     y_test = pd.Series(y_test.squeeze()).map(dict_classes)
#     # y_pred = pd.Series(np.array(pred_pos).astype(int)).map(dict_classes)
#     y_pred = pd.Series(np.array(pred_pos, dtype=np.int)).map(dict_classes)
    
#     confusion_matrix = pd.crosstab(y_test,
#                                    y_pred,
#                                    rownames=['Actual'], 
#                                    colnames=['Predicted']
#                                    )
#     print(confusion_matrix)
  
#     # compute FPR and TPR
#     # tp = confusion_matrix.iloc[0,0]
#     # fp = confusion_matrix.iloc[1,0]
#     # fn = confusion_matrix.iloc[0,1]
#     # tn = confusion_matrix.iloc[1,1]
    
#     # TPR = tp / (tp + fn)
#     # FPR = fp/ (fp + tn)
    
#     # print(f"Threshold default |  CD69- : {tn + fn} | CD69+ {tp + fp} | TPR: {TPR:.3f} FPR: {FPR:.3f}")
    
#     # TODO fix this list
#     # for col, feature in zip(np.flip(df.columns[np.argsort(clf.feature_importances_)]), 
#     #                         np.flip(np.argsort(clf.feature_importances_))):
#     #     print(col, clf.feature_importances_[feature])

#     #print metrics to assess classifier performance
#     accuracy = accuracy_score(y_test, y_pred)
#     print('Accuracy score =', accuracy)
#     print(classification_report(y_test, y_pred))

#     return fpr, tpr, roc_auc, accuracy, (FPR, TPR) # (operating_point_fpr, operating_point_tpr)#
