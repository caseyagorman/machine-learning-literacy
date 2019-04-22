import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneOut

import matplotlib.pyplot as plt

# data_path = './data/reading-levels.csv'
data_path = './data/ln-ls-data.csv'
columns_to_skip = 0

def get_training_data(path):
    columns = get_column_names(path)
    number_of_columns = len(columns)

    loaded_text = np.loadtxt(
        path, delimiter=",", skiprows=1,
        usecols=range(columns_to_skip, number_of_columns)
    )
    print('loaded_text', loaded_text)
    X, y = np.hsplit(loaded_text,  [-1])
    y = y.flatten()
    return X, y

def get_column_names(path):
    with open(path) as fp:
        header = fp.readline().split(',')#[1:-1]
    return header

X, y = get_training_data(data_path)


# Split our training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
    )

regr = linear_model.LinearRegression()
# clf = linear_model.Lasso(alpha=0.1)

# # Scale our data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
mean_squared_error = mean_squared_error(y_test, y_pred)
variance = r2_score(y_test, y_pred)
print('mean_squared_error', mean_squared_error)
print('variance', variance)

# Do some classification. Set reading levels less than 4 to not passing.
# y = [0 if n < 4.0 else 1 for n in y]
# y = np.array(y)

loo = LeaveOneOut()
loo.get_n_splits(X)

scores = []
rfc_true = []
rfc_pred = []
rfc_proba = []

for train_index, test_index in loo.split(X):
    rfc = ensemble.RandomForestClassifier(n_estimators=60)
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    rfc_true += [y_test[0]]
    rfc_pred += [y_pred[0]]
    y_proba = rfc.predict_proba(X_test)[0][1]
    rfc_proba += [y_proba]
    # print(X_train, X_test, y_train, y_test)

print('rf classification_report')
print(classification_report(rfc_true, rfc_pred))

logr_true = []
logr_pred = []
logr_proba = []

for train_index, test_index in loo.split(X):
    logr = linear_model.LogisticRegression()
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    logr.fit(X_train, y_train)
    y_pred = logr.predict(X_test)
    logr_true += [y_test[0]]
    logr_pred += [y_pred[0]]
    y_proba = logr.predict_proba(X_test)[0][1]
    logr_proba += [y_proba]

print('logr classification_report')
print(classification_report(logr_true, logr_pred))


# rfc.fit(X_train, y_train)
#
# feature_names = get_column_names(data_path)[:-1]
# # Plot the feature importances of the forest
# # plt.figure()
# plt.title("Feature importances")
# plt.bar(feature_names, rfc.feature_importances_,
#        color="r", align="center", )
# plt.xticks(rotation=60)
#
# plt.show()
