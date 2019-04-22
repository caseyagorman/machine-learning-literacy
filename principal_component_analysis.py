import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Binarizer, MinMaxScaler
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

from scipy import linalg

import matplotlib.pyplot as plt

data_path = './data/reading-levels.csv'
# data_path = './data/ln-ls-data.csv'
columns_to_skip = 2
reading_datasets = {
    'reading_levels': {
        'data_path': './data/reading-levels.csv',
        'columns_to_skip': 2,
        'columns_to_use': range(2,28),
        'n_features': 25
    },
    'ln_ls_data': {
        'data_path': './data/ln-ls-data.csv',
        'columns_to_skip': 0,
        'columns_to_use': range(3),
        'n_features': 2
    }
}

def get_training_data(dataset):
    data_path = dataset['data_path']
    columns_to_use = dataset['columns_to_use']
    # columns = get_column_names(path)
    # number_of_columns = len(columns)

    loaded_text = np.loadtxt(
        data_path, delimiter=",", skiprows=1,
        usecols=columns_to_use
    )
    print('loaded_text', loaded_text)
    X, y = np.hsplit(loaded_text,  [-1])
    y = y.flatten()
    return X, y

def get_column_names(path):
    with open(path) as fp:
        header = fp.readline().split(',')#[1:-1]
    return header

dataset = reading_datasets['reading_levels']
data_path = dataset['data_path']
X, y = get_training_data(dataset)

# Binarize labels
# EOY fail threshold
fail_threshold = 2
y = Binarizer(threshold=fail_threshold).transform([y])[0]
target_names = get_column_names(data_path)[dataset['columns_to_use'][0]:dataset['columns_to_use'][-1]]
# Calculate covariance between all variables and EOY reading levels
covar = np.cov(X, y, rowvar=False)
plt.bar(np.arange(len(covar[-1][:-1])), covar[-1][:-1], tick_label=target_names)
plt.xticks(rotation=60)
plt.title("Covariance, Unscaled")
plt.figure()

X_scaled = MinMaxScaler().fit_transform(X)
# Calculate covariance between all variables and EOY reading levels
covar = np.cov(X_scaled, y, rowvar=False)
plt.bar(np.arange(len(covar[-1][:-1])), covar[-1][:-1], tick_label=target_names)
plt.xticks(rotation=60)
plt.title("Covariance, Scaled")
# plt.figure()

# Calculate variance
plt.figure()
var = np.var(X, axis=0)
plt.bar(np.arange(len(var)), var, tick_label=target_names)
plt.xticks(rotation=60)
plt.title("Variance, Unscaled")

# Calculate scaled variance
plt.figure()
var = np.var(X_scaled, axis=0)
plt.bar(np.arange(len(var)), var, tick_label=target_names)
plt.xticks(rotation=60)
plt.title("Variance, Scaled")

# Linear discriminant analysis!
# lda = LinearDiscriminantAnalysis(store_covariance=True).fit(X_scaled, y)

explained_variance_ratios = {}
for n_components in range(25, 0, -1):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_scaled, y)
    explained_variance_ratios[n_components] = lda.explained_variance_ratio_

# Calculate mutual information
plt.figure()
mutual_info = mutual_info_classif(X, y, True)
plt.bar(np.arange(len(var)), mutual_info, tick_label=target_names)
plt.xticks(rotation=60)
plt.title("Mutual Info")
# plt.show()
# var_scaled = np.var(X_scaled, axis=1)

def score_classification(clf, X, y):
    # Split our training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
        )
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

# quick logit score
regr = linear_model.LinearRegression()
X = X_scaled
print(score_classification(linear_model.LogisticRegression(), X, y))

lda = LinearDiscriminantAnalysis(n_components=10)
X = lda.fit_transform(X, y)

# # Do some dimensionality reduction!
# sel = VarianceThreshold(threshold=0.1)
# X = sel.fit_transform(X_scaled)
# feature_names = sel.transform([target_names])

print(score_classification(linear_model.LogisticRegression(), X, y))
#
# n_features = X.shape[1]
#
# n_components = np.arange(0, n_features, 5)  # options for n_components
#
# def compute_scores(X):
#     pca = PCA(svd_solver='full')
#     fa = FactorAnalysis()
#
#     pca_scores, fa_scores = [], []
#     for n in n_components:
#         pca.n_components = n
#         fa.n_components = n
#         pca_scores.append(np.mean(cross_val_score(pca, X)))
#         fa_scores.append(np.mean(cross_val_score(fa, X)))
#
#     return pca_scores, fa_scores
#
#
# def shrunk_cov_score(X):
#     shrinkages = np.logspace(-2, 0, 30)
#     cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
#     return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))
#
#
# def lw_score(X):
#     return np.mean(cross_val_score(LedoitWolf(), X))
#
#
# for X, title in [(X, 'Homoscedastic Noise')]:
#     pca_scores, fa_scores = compute_scores(X)
#     n_components_pca = n_components[np.argmax(pca_scores)]
#     n_components_fa = n_components[np.argmax(fa_scores)]
#
#     pca = PCA(svd_solver='full', n_components='mle')
#     pca.fit(X)
#     n_components_pca_mle = pca.n_components_
#
#     print("best n_components by PCA CV = %d" % n_components_pca)
#     print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
#     print("best n_components by PCA MLE = %d" % n_components_pca_mle)
#
#     plt.figure()
#     plt.plot(n_components, pca_scores, 'b', label='PCA scores')
#     plt.plot(n_components, fa_scores, 'r', label='FA scores')
#     # plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
#     plt.axvline(n_components_pca, color='b',
#                 label='PCA CV: %d' % n_components_pca, linestyle='--')
#     plt.axvline(n_components_fa, color='r',
#                 label='FactorAnalysis CV: %d' % n_components_fa,
#                 linestyle='--')
#     plt.axvline(n_components_pca_mle, color='k',
#                 label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')
#
#     # compare with other covariance estimators
#     plt.axhline(shrunk_cov_score(X), color='violet',
#                 label='Shrunk Covariance MLE', linestyle='-.')
#     plt.axhline(lw_score(X), color='orange',
#                 label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')
#
#     plt.xlabel('nb of components')
#     plt.ylabel('CV scores')
#     plt.legend(loc='lower right')
#     plt.title(title)

# plt.show()
