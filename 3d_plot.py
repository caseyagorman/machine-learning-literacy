import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits.mplot3d import Axes3D

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
fail_threshold = 1.0
y = Binarizer(threshold=fail_threshold).transform(y.reshape(-1, 1))

# Split our training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
    )

ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)
# #############################################################################
# Plot the figure
def plot_figs(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')
    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
                    np.array([[-.1, .15], [-.1, .15]]),
                    clf.predict(np.array([[-.1, -.1, .15, .15],
                                          [-.1, .15, -.1, .15]]).T
                                ).reshape((2, 2)),
                    alpha=.5)
    ax.set_xlabel('Letter Names')
    ax.set_ylabel('Letter Sounds')
    ax.set_zlabel('F&P Level')
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])

#Generate the three different figures from different views
elev = 43.5
azim = -110
plot_figs(1, elev, azim, X_train, ols)

# elev = -.5
# azim = 0
# plot_figs(2, elev, azim, X_train, ols)

# elev = -.5
# azim = 90
# plot_figs(3, elev, azim, X_train, ols)

plt.show()
