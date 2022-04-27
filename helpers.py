import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()
import altair as alt
from altair.vega import v5
from IPython.display import HTML
import json
import graphviz
from sklearn import tree
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier



def feature_eng(num_col,cat_col,X,y):

    numerical_columns=num_col
    categorical_columns=cat_col
    categorical_pipeline= Pipeline([
        ("cat_imputer",SimpleImputer(strategy="most_frequent")),
        ("cat_ohe",OrdinalEncoder())
    ])

    numerical_pipeline= Pipeline([
        ("cat_imputer",SimpleImputer(strategy="most_frequent")),
        ("cat_ohe",MinMaxScaler())

    ])
    column_transformer=ColumnTransformer([
        ("sex_imputer",categorical_pipeline,categorical_columns),
        ("num_scaler",numerical_pipeline,numerical_columns)
    ])
    column_transformer.fit(X)
    X_fe =column_transformer.transform(X)

    # Synthetic minority oversampling technique, outputs numpy array
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_fe, y)

    return X_res,y_res


def evaluation(ypred, ytest, model):
    cm = confusion_matrix(ytest, ypred)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    false_pos_rate = FP / (TN + FP)
    false_neg_rate = FN / (TP + FN)
    F1_score=(2*precision*recall)/(precision+recall)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'False positive rate: {false_pos_rate}')
    print(f'False negative rate: {false_neg_rate}')
    print(f'F1 score: {F1_score}')
    f = open("models.txt", "a+")
    f.write(
        f'{model}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nFalse positive rate: '
        f'{false_pos_rate}\nFalse negative rate: {false_neg_rate}\nAccuracy: {accuracy}\n\n')
    return false_pos_rate, false_neg_rate

def feature_eng_2(num_col,cat_col,X,y):

    numerical_columns=num_col
    categorical_columns=cat_col
    categorical_pipeline= Pipeline([
        ("cat_imputer",SimpleImputer(strategy="most_frequent")),
        ("cat_ohe",OrdinalEncoder())
    ])

    # bining_pipeline= Pipeline([
    #     ("cat_imputer",SimpleImputer(strategy="mean")),
    #     ("cat_ohe",KBinsDiscretizer(n_bins=4, encode='onehot-dense', strategy='quantile'))

    # ])

    numerical_pipeline= Pipeline([
        ("cat_imputer",SimpleImputer(strategy="most_frequent")),
        ("cat_ohe",MinMaxScaler())

    ])
    column_transformer=ColumnTransformer([
        ("sex_imputer",categorical_pipeline,categorical_columns),
        ("num_scaler",numerical_pipeline,numerical_columns)
    ])
    column_transformer.fit(X)
    X_fe =column_transformer.transform(X)

    # Synthetic minority oversampling technique, outputs numpy array
    rus =SMOTE(sampling_strategy=0.1, random_state=12)
    X_res_2, y_res_2= rus.fit_resample(X_fe, y)

    return X_res_2,y_res_2

def plot_roc(model, model_name,y_test,X_test):
    fpr, tpr, threshold = roc_curve(y_test, model.predict(X_test))
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'plots/roc_{model_name}.png')
    return plt.show()

def logreg(xtrain, xtest, ytrain, ytest):
    """ Logistic regression """
    model_name = 'Logistic Regression'
    lr = LogisticRegression(random_state=12)
    lr.fit(xtrain, ytrain)
    pred = prediction(lr, xtest)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, model_name)
    plot_roc(lr, model_name,ytest,xtest)
    return lr, false_pos, false_neg

def conf_matrix(ytest, ypred, model_name):
    plt.figure()
    cm = confusion_matrix(ytest, ypred)
    labels = ['0', '1']
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', vmin=0.2, cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f'plots/confusion_matrix_{model_name}.png')
    plt.show()
    return cm

def prediction(model, xtest):
    predict = model.predict(xtest)
    return predict

def randforest(xtrain, xtest, ytrain, ytest):
    """ Random Forest Classifier"""
    model_name = 'Random Rorest'
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=12, max_depth=100)
    clf_rf.fit(xtrain, ytrain)
    pred = prediction(clf_rf, xtest)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, 'Random Rorest')
    plot_roc(clf_rf, model_name,ytest,xtest)
    return clf_rf, false_pos, false_neg

def xgboost(xtrain, xtest, ytrain, ytest ):
    """ XGBoost """
    model_name = 'XGBoost'
    xgb = XGBClassifier(n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        missing=-999,
        random_state=2019,
        tree_method='auto',
        n_jobs = -1)
    xgb.fit(xtrain, ytrain)
    pred = prediction(xgb, xtest)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, model_name)
    plot_roc(xgb, model_name,ytest,xtest)
    return xgb, false_pos, false_neg

def lightgb(xtrain, xtest, ytrain, ytest):
    """ Light Gradient Boosting """
    model_name = 'LightGB'
    gbm = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.06, class_weight={0: 1, 1: 7})
    gbm.fit(xtrain, ytrain, eval_set=(xtest, ytest))
    pred = prediction(gbm, xtest)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, model_name)
    plot_roc(gbm, model_name,ytest,xtest)
    return gbm, false_pos, false_neg

