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

data_path = './data/ERD_MASTER_DATA.csv'

df = pd.read_csv(data_path)

features_for_feature_importance = [
    ''
]
cols_to_drop = [
    'School', 'Grade Level', 'First Name', 'Last Name', 'Teacher',
    'Blending Q2', 'Final Sound Q2', 'Segmenting Q2', 'SWC Q2',
    # 'F&P Q2'
    # 'Hearing Rhymes Q2', 'Creating Rhymes Q2', 'Initial Sound Q2',
    # 'Blending Q2', 'Segmenting Q2'
]
target_name = ['F&P Q2']
# cols_to_onehot = ['Teacher']
# Cols to binarize should be a list of (column_name, threshold)
def preprocess_data(
        df, cols_to_drop=[], cols_to_onehot=[], scale=False, scaler=StandardScaler(),
        cols_to_binarize=[]
        ):
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
y = binarize(y, 2)

# print(f_classif(X, y))
#
def get_feature_importances(X, y, importance_method='rf'):
    # X = X.drop(columns=additional_cols_to_drop)

    # selector = f_classif(X, y)
    # selector.fit(X, y)
    # scores = -np.log10(f_classif(X, y)[0])
    # return scores / max(scores)
    # X = X.drop(columns=additional_cols_to_drop)
    if importance_method == 'lda':
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        scores = lda.coef_[0]
        importances = scores
    elif importance_method == 'rf':
        # return scores
        clf = ExtraTreesClassifier(n_estimators=500)
        clf.fit(X, y)
        importances = clf.feature_importances_
    elif importance_method == 'mutual_info':
        importances = feature_selection.mutual_info_classif(X, y, True)
    # Standardize importances
    importances = (importances-importances.min()) / (importances.max()-importances.min())
    # importances = StandardScaler().fit_transform([importances])
    # return pd.DataFrame(list(zip(X.columns, importances)), columns=['name', 'importance'])\
    importances = pd.DataFrame({'importance': pd.Series(importances, index=X.columns)})

    importances = importances.sort_values('importance', ascending=False)
    importances.importance_method = importance_method
    return importances
    # return importances.sort_values('importance', ascending=False)
        # .sort_values('importance', ascending=False)
        # .set_index('name') \
    # model = SelectFromModel(clf, prefit=False)
    # model.
    # return model

# def plot_feature_importances(X, y, additional_cols_to_drop=[]):
def plot_feature_importances(feature_importances):
    # X = X.drop(columns=additional_cols_to_drop)
    # feature_importances = get_feature_importances(X, y)
    ax = feature_importances.plot.bar(
        title='Feature Importances, {}'.format(feature_importances.importance_method), rot=60
    )
    # plt.bar(np.arange(X.shape[1]), feature_importances, tick_label=X.columns)
    # ax.set_xticklabels(rotation=60)
    # ax.title("Feature Importances")
    # plt.show()

feature_importances = get_feature_importances(X, y, 'rf')
plot_feature_importances(feature_importances)
feature_importances = get_feature_importances(X, y, 'lda')
plot_feature_importances(feature_importances)
feature_importances = get_feature_importances(X, y, 'mutual_info')
plot_feature_importances(feature_importances)
plt.show()
# Just Q2
def get_sight_words(df, data_cols, target_col, threshold):
    X = df[data_cols].sum(axis=1)
    y = df[target_col]
    y = binarize(y, threshold)
    quarter = target_col.split(" ")[-1]
    X.X_label = "Sight Words, "+ quarter
    X.title = "Sight Words, "+ quarter + ' vs. ' + target_col
    return X, y, target_col

def get_sight_words_test_values(X):
    return np.linspace(0, X.max() + 1, 300)

def get_ln_ls(df, data_cols, target_col, threshold):
    X = df[data_cols].T.squeeze()
    y = df[target_col]
    y = binarize(y, threshold)
    quarter = data_cols[0].split(" ")[-1]
    # print('target_col', target_col)
    if "LN" in data_cols[0]:
        data_type = "Letter Names, "
    else:
        data_type = "Letter Sounds, "
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

# X_, y_ = get_sight_words(df, ['SWA Q2', 'SWB Q2'], 'F&P Q2', 1)
plot_logit(*get_sight_words(df, ['SWA Q2', 'SWB Q2'], 'F&P Q2', 1))

plot_logit(*get_sight_words(df, ['SWA Q3', 'SWB Q3', 'SWC Q3'], 'F&P Q3', 2))
plot_logit(*get_ln_ls(df, ['LN Q1'], 'F&P Q3', 2))
plot_logit(*get_ln_ls(df, ['LN Q2'], 'F&P Q3', 2))
plot_logit(*get_ln_ls(df, ['LN Q3'], 'F&P Q3', 2))

plot_logit(*get_ln_ls(df, ['LS Q1'], 'F&P Q3', 2))
plot_logit(*get_ln_ls(df, ['LS Q2'], 'F&P Q3', 2))
plot_logit(*get_ln_ls(df, ['LS Q3'], 'F&P Q3', 2))
plt.show()
plot_logit( X['LN Q1'], y[''] )
