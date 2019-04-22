import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# data_path = './data/reading-levels.csv'
data_path = './data/ln-ls-data.csv'
columns_to_skip = 0
dependent_variable_name = 'F & P Level'
fail_threshold = 1.0

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
letter_names = X[:, 0].reshape(-1, 1)
letter_sounds = X[:, 1].reshape(-1, 1)

# Binarize labels
y = Binarizer(threshold=fail_threshold).transform(y.reshape(-1, 1))

datasets = [
    ('letter names', letter_names, (0, 13, 26, 39, 52)),
    # ('letter sounds', letter_sounds, (0, 13, 26))
]

for independent_variable_name, X_data, X_ticks in datasets:

    # Create linear regression object
    regr = linear_model.LinearRegression(normalize=True)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(X_data, y,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.title('{} vs {}'.format(independent_variable_name, dependent_variable_name))

    plt.ylabel(dependent_variable_name)
    plt.xlabel(independent_variable_name)

    plt.xticks(X_ticks)
    # plt.yticks(())

    plt.show()
