import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Binarizer, RobustScaler
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt
# plt.clf()

h = .02  # step size in the mesh

x_label_text = 'Letter Names'
y_label_text = 'Letter Sounds'

school_names = [
    'EPACS',
    'BMA',
    'Monarch'
]

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    # "Decision Tree",
    "Random Forest",
    "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA"
]


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
]

# data_path = './data/reading-levels.csv'
data_path = './data/ln-ls-w-school.csv'
columns_to_skip = 0
dependent_variable_name = 'F & P Level'
fail_threshold = 1.0

def get_training_data(path, school_id=None):
    columns = get_column_names(path)
    number_of_columns = len(columns)

    loaded_text = np.loadtxt(
        path, delimiter=",", skiprows=1,
        usecols=range(columns_to_skip, number_of_columns)
    )
    if school_id != None:
        loaded_text = loaded_text[loaded_text[:, 0] == school_id][:,1:]

    # print('loaded_text', loaded_text)
    X, y = np.hsplit(loaded_text,  [-1])
    y = y.flatten()
    return X, y

def get_column_names(path):
    with open(path) as fp:
        header = fp.readline().split(',')#[1:-1]
    return header

for school_id, school_name in enumerate(school_names):
    X, y = get_training_data(data_path, school_id)
    letter_names = X[:, 0].reshape(-1, 1)
    letter_sounds = X[:, 1].reshape(-1, 1)

    # Binarize labels
    y = Binarizer(threshold=fail_threshold).transform(y.reshape(1, -1))[0]

    reading_data = (X, y)

    datasets = [
        reading_data
    ]

    # points where we want ticks, as well as the label for that tick
    ticks = [
        [0, 0],
        [13, 7],
        [26, 13],
        [39, 20],
        [52, 26]
    ]
    ticks = np.array(ticks)

    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    # LN is X, LS is Y
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        scaler = StandardScaler().fit([[0,0], [52, 26]])
        # scaler = RobustScaler()
        X = scaler.transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        # x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        # y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        x_min, x_max = 0 - 0.1, 52 + 0.1
        y_min, y_max = 0 - 0.1, 26 + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(111)
        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        scaled_ticks = scaler.transform(ticks)
        # print(scaled_ticks)
        # print(scaled_ticks[:, 0])
        ax.set_xticks(scaled_ticks[:, 0])
        ax.set_xticklabels(ticks[:, 0])
        ax.set_yticks(scaled_ticks[:, 1])
        ax.set_yticklabels(ticks[:, 1])
        ax.set_ylabel(y_label_text)
        ax.set_xlabel(x_label_text)
        # ax.set_yticks(scaled_ticks[:, 1], ticks[:, 1])
        # ax.set_xticks(scaled_ticks[:, 0], ticks[:, 0])
        # ax.set_yticks(scaled_ticks[:, 1], ticks[:, 1])
        # print(scaler.transform(x_ticks))
        # ax.set_xticks(scaler.transform(x_ticks).reshape(1, -1)[0], x_ticks)
        # ax.set_yticks(scaler.transform(y_ticks).reshape(1, -1)[0], y_ticks)
        # ax.set_yticks(())
        i += 1
        #
        # iterate over classifiers
        for clf_name, clf in zip(names, classifiers):
            ax = plt.figure().add_subplot(111)
            # .add_subplot(111)
            # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print(score)
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(scaled_ticks[:, 0])
            ax.set_xticklabels(ticks[:, 0])
            ax.set_yticks(scaled_ticks[:, 1])
            ax.set_yticklabels(ticks[:, 1])
            ax.set_ylabel(y_label_text)
            ax.set_xlabel(x_label_text)
            ax.set_title('{}, {}'.format(school_name, clf_name))
            # if ds_cnt == 0:
            #     ax.set_title(name)
            # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            #         size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()
