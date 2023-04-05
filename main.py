"""
Created on Wed Mar  8 12:04:48 2023
@author: Ahmad Al Musawi
"""

from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

def get_classification_metrics(y_true, y_pred):
    # compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # compute sensitivity and specificity for each class
    num_classes = cm.shape[0]
    sensitivity = []
    specificity = []
    for i in range(num_classes):
        tp = cm[i,i]
        fn = np.sum(cm[i,:]) - tp
        fp = np.sum(cm[:,i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        sensitivity_i = tp / (tp + fn)
        specificity_i = tn / (tn + fp)
        sensitivity.append(sensitivity_i)
        specificity.append(specificity_i)
    
    # compute macro-average sensitivity and specificity
    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    
    return accuracy, macro_sensitivity, macro_specificity

def get_classification_metrics1(y_true, y_pred):
    # compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    tn, fp, fn, tp = cm[0], cm[1], cm[2], cm[3]
    
    # compute sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(sensitivity)
    # # compute AUC-ROC
    # auc_roc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    
    # # compute AUC-PR
    # precision, recall, _ = precision_recall_curve(y_true, y_pred)
    # auc_pr = average_precision_score(y_true, y_pred)
    
    return accuracy, sensitivity, specificity#, auc_roc, auc_pr

def preprocessing(df):
    print('preprocessing...')
    return df

def split_labels(df, cols):
    '''split the dataframe into predicting table and labels
       df: given dataset
       cols: list of labels
    '''
    return df[[i for i in df if i not in cols]], df[cols]
    
def LinearSVM(X_train,y_train, X_test):
    svm = LinearSVC(random_state=42)
    svm.fit(X_train, y_train)
    
    # make predictions on the test set
    y_pred = svm.predict(X_test)
    return y_pred

def GaussianSVM(X_train,y_train, X_test):
    print('implementing SVM...')
    clf = SVC(kernel='rbf', C=1.0) # Gaussian radial basis function (RBF) kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def PolySVM(X_train,y_train, X_test):
    print('implementing SVM...')
    clf = SVC(kernel='poly', degree=2, coef0=1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def SigmoidSVM(X_train,y_train, X_test):
    print('implementing SVM...')
    clf = SVC(kernel='sigmoid', gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def NaiveBayes(X_train,y_train, X_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    return y_pred

def Logistic(X_train,y_train, X_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    # make predictions on the test set
    y_pred = lr.predict(X_test)    
    return y_pred

def CART(X_train,y_train, X_test):
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def PCA_model(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    df_pca = pd.DataFrame(data=X_pca)
    print(df_pca.shape)
    return df_pca

def CE_Model(X):
    embedding = SpectralEmbedding(n_components=2)
    X_CE = embedding.fit_transform(X)
    
    print(X_CE.shape)
    return X_CE

def CFS(X, y):
    selector = SelectKBest(score_func=f_regression, k=5)
    X_new = selector.fit_transform(X, y)
    return X_new

def LLCFS(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def ILFS(X, y):
    # create a linear regression model
    model = LinearRegression()
    
    # define the search space
    k_features = np.arange(1, X.shape[1]+1)
    
    # create a sequential feature selector object
    selector = SequentialFeatureSelector(model, k_features=k_features, forward=True, scoring='r2', cv=5)
    
    # perform incremental feature selection
    selector.fit(X, y)
    
    # print the selected feature indices
    print("Indices of selected features:", selector.k_feature_idx_)

def one_split(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 

# Load the text file into a DataFrame
# df1 = pd.read_csv('processed.cleveland.data', delimiter=',', header=None)
df1 = pd.read_excel('cleveland data.xlsx')
df2 = pd.read_excel('CTG.xls', sheet_name = 'Raw Data')

df2 = df2[[i for i in df2 if i not in ['FileName','Date','SegFile']]]

X1, Y1 = split_labels(df1, ['num'])
X2, Y2 = split_labels(df2, ['CLASS'])

X1_train, X1_test, y1_train, y1_test = one_split(X1, Y1)
X2_train, X2_test, y2_train, y2_test = one_split(X2, Y2)


# performing experiment 1
predictors = [LinearSVM, GaussianSVM, PolySVM, SigmoidSVM, NaiveBayes, Logistic, CART]
# predictors = [LinearSVM]#, GaussianSVM, PolySVM, SigmoidSVM, NaiveBayes, Logistic]


pred_Y1 = [pred(X1_train,y1_train, X1_test) for pred in predictors]
pred_Y2 = [pred(X2_train,y2_train, X2_test) for pred in predictors]


results1 = [get_classification_metrics(y1_test, p) for p in pred_Y1]
results2 = [get_classification_metrics(y2_test, p) for p in pred_Y2]




# # performing experiment 2    
# df1 = stats.zscore(X1)
# df2 = stats.zscore(X2)




