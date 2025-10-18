from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# RandomizedSearchCV global config
RANDOM_SEARCH_PARAMS = {
    "n_iter": 20,
    "cv": 3,
    "n_jobs": -1,
    "verbose": 2,
    "random_state": 42,
    "scoring": "accuracy"
}


PARAMS = [

    # ---------------------- Logistic Regression ----------------------
    ('Logistic Regression', LogisticRegression(), [
        {
            'solver': ['lbfgs', 'newton-cg', 'sag'],
            'penalty': ['l2', 'none'],
            'C': [0.1, 0.3, 0.5, 0.7, 1.0],
            'max_iter': [100, 300, 500, 700]
        },
        {
            'solver': ['saga'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'l1_ratio': [0.0, 0.5, 1.0],
            'C': [0.1, 0.3, 0.5, 0.7, 1.0],
            'max_iter': [100, 300, 500, 700]
        }
    ]),

    # ---------------------- Decision Tree ----------------------
    ('Decision Tree Classifier', DecisionTreeClassifier(), {
        'criterion': ['gini', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 3, 5, 7, 10],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }),

    # ---------------------- Random Forest ----------------------
    ('Random Forest Classifier', RandomForestClassifier(), {
        'n_estimators': [100, 200, 500, 800],
        'criterion': ['gini', 'log_loss'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }),

    # ---------------------- Gradient Boosting ----------------------
    ('Gradient Boosting Classifier', GradientBoostingClassifier(), {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.6, 0.8, 1.0],
        'criterion': ['friedman_mse', 'squared_error'],
        'max_features': ['sqrt', 'log2']
    }),

    # ---------------------- K Nearest Neighbors ----------------------
    ('K Nearest Neighbors Classifier', KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'leaf_size': [20, 40, 60],
        'p': [1, 2]
    }),

    # ---------------------- Extra Trees ----------------------
    ('Extra Trees Classifier', ExtraTreesClassifier(), {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'log_loss'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }),

    # ---------------------- AdaBoost ----------------------
    ('AdaBoost Classifier', AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }),

    # ---------------------- CatBoost ----------------------
    ('CatBoost Classifier', CatBoostClassifier(
        eval_metric="F1",
        loss_function="Logloss",
        verbose=0
    ), {
        "iterations": [200, 400, 600],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
        "border_count": [32, 64, 128],
        "bagging_temperature": [0, 0.5, 1, 2],
        "random_strength": [0.5, 1, 2],
        "class_weights": [[1, 5], [1, 10], [1, 20]]
    }),

    # ---------------------- XGBoost ----------------------
    ('XGB Classifier', XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ), {
        "n_estimators": [200, 500, 1000],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_lambda": [1, 5, 10],
        "reg_alpha": [0, 0.1, 0.5, 1],
        "scale_pos_weight": [1, 5, 10]
    })
]
