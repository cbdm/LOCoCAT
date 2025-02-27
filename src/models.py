from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Define the models we want to train.
models = {}

# KNN
# Adapted from https://towardsdatascience.com/gridsearchcv-for-beginners-db48a90114ee
models["knn"] = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "leaf_size": [15, 20],
    },
    scoring="accuracy",
    cv=5,
)

# DecisionTree
# Params chosen from: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
models["tree"] = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid={
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [2, 3, 5, 8, 10, None],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "random_state": [0],
    },
    scoring="accuracy",
    cv=5,
)

# RandomForest
# Params chosen from: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
models["forest"] = GridSearchCV(
    RandomForestClassifier(),
    param_grid={
        "n_estimators": [1, 2, 5, 10, 20],
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 5, None],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "random_state": [0],
    },
    scoring="accuracy",
    cv=5,
)

# AdaBoost
# Params chosen from: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
models["adaboost"] = GridSearchCV(
    AdaBoostClassifier(),
    param_grid={
        "n_estimators": [1, 2, 5, 10, 20, 50],
        "learning_rate": [0.1, 0.3, 0.5, 0.7, 1.0],
        "algorithm": ["SAMME", "SAMME.R"],
        "random_state": [0],
    },
    scoring="accuracy",
    cv=5,
)

# GaussianBayes
# Params chosen from: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
models["gaussiannb"] = GridSearchCV(
    GaussianNB(),
    param_grid={
        "priors": [None],
    },
    scoring="accuracy",
    cv=5,
)

# QuadraticDiscriminantAnalysis
# Params chosen from: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
models["qda"] = GridSearchCV(
    QuadraticDiscriminantAnalysis(),
    param_grid={
        "priors": [None],
    },
    scoring="accuracy",
    cv=5,
)
