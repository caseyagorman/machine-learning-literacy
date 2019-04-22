import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Binarizer, RobustScaler,\
    OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.feature_selection import f_classif, SelectFromModel
from sklearn import feature_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pandas as pd

import matplotlib.pyplot as plt
# plt.clf()

data_path = './data/Grade2DATAGRID.csv'
# file to use

df = pd.read_csv(data_path)
# variable for getting the path to the file

features_for_feature_importance = [
    ''
]

# ?
# This didn't drop?

target_name = ['F&P2']
# The thing I want to find out about
# Cols to binarize should be a list of (column_name, threshold)
def preprocess_data(
        df, , scale=False, scaler=StandardScaler(),
        cols_to_binarize=[]
        ):

# datafile, columns to drop,
    # df = df.drop(columns=cols_to_drop)
    # for col_name in cols_to_onehot:
    #     le = LabelEncoder()
    #     col_data = le.fit_transform(df[col_name])
    #     # encoder = OneHotEncoder()
    #     # col_data = encoder.fit_transform([col_data])
    #     df[col_name] = col_data
    df = df.dropna()
    return df
    # df[binarize_cols] = df[binarize_cols].apply(to_date)

def split_xy(df, target_name):
    y = df[target_name]
    X = df.drop(columns=target_name)
    return X, np.ravel(y)

def standardize(df):
    std = MinMaxScaler()
    data = std.fit_transform(df)
    # return data
    return pd.DataFrame.from_records(data, columns=df.columns)

def binarize(y, threshold):
    return Binarizer(threshold=threshold).transform([y])[0]
    # y = Binarizer(threshold=threshold).transform(y.reshape(1, -1))[0]

df = preprocess_data(df, cols_to_drop=cols_to_drop)
X, y = split_xy(df, target_name)
X = standardize(X)
y = binarize(y, 274)


def get_STAR1(df, data_cols, target_col, threshold):
    X = df[data_cols].sum(axis=1)
    y = df[target_col]
    y = binarize(y, threshold)
    # quarter = target_col.split(" ")[-1]
    X.X_label = "STAR 1"
    X.title = "STAR 1" 'vs.' + target_col
    return X, y, target_col

def get_STAR1_test_values(X):
    return np.linspace(0, X.max() + 1, 300)

def get_FP2(df, data_cols, target_col, threshold):
    X = df[data_cols].T.squeeze()
    y = df[target_col]
    y = binarize(y, threshold)
    quarter = data_cols[0].split(" ")[-1]
    # print('target_col', target_col)
    if "F&P 2" in data_cols[0]:
        data_type = "F&P 2, "

    X.X_label = data_type + quarter
    X.title = data_cols[0] + " vs. " + target_col
    return X, y, target_col

# letter names is X, with max of 52
def get_ln_ls_test_values(X):
    ln = np.linspace(0, 53, 300)
    ls = np.linspace(0, 27, 300)
    return list(zip(ln, ls))

def plot_logit(X, y, target_col):
    X_label = X.X_label
    title = X.title
    print('X_label', X_label, 'target_col', target_col)
    X = X.reshape(-1, 1)

    # X_train, X_test, y_train, y_test = \
    #     train_test_split(X, y, test_size=.2, random_state=42)
    clf = LogisticRegression(C=1e5)
    clf.fit(X, y)

    # and plot the result
    # plt.figure(1, figsize=(4, 3))
    # plt.clf()
    plt.figure()
    plt.scatter(X.ravel(), y, color='black', zorder=20)
    # test_values = get_test_values(X)
    test_values = np.linspace(0, X.max() + 1, 300)
    def model(x):
        return 1 / (1 + np.exp(-x))
    loss = model(test_values * clf.coef_ + clf.intercept_).ravel()
    plt.plot(test_values, loss, color='red', linewidth=3)

    plt.axhline(.5, color='.5')

    plt.ylabel('Proba of Passing')
    plt.xlabel('Number of ' + X_label)
    plt.title(title)
    # plt.xticks(range(-5, 10))
    # plt.yticks([0, 0.5, 1])
    plt.ylim(-.1, 1.1)
    plt.xlim(-1, X.max() + 1)
    # plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
    #            loc="upper left", fontsize='small')
    # plt.show()

plt.show()
# plot_logit( X['LN Q1'], y[''] )
