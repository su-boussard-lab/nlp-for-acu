""" This file contains the training script for the Bayesian logistic regression models
Author:
    Claudio Fanconi
"""
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.utils.config import config
from src.data_preprocessing import preprocessing


def cv_fit_predict_LASSO(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    label_type: str,
    mode: str,
    random_state: int,
) -> None:
    """Grid-search CV a logistic regression model, with the train data and predict on the test data.
    Saves predicitons in npz file.
    Args:
        X_train (np.ndarray): normalized training features
        y_train (np.ndarray): training labels
        X_test (np.ndarray): normalized testing features
        label_type (str): type of time (30, 180, 365d) of the ACU prediciton label
        mode (str, "tabular"): mode of the features preprocessed. either `tabular`, `language` or `fusion`
        random_state (int): random state for reproducibility
    returns:
        None
    """
    y_train = y_train[label_type]
    assert mode.lower() in ["tabular", "language", "fusion"]
    print(f"Starting grid search for {mode} model on {label_type} labels.")
    clf = GridSearchCV(
        estimator=LogisticRegression(penalty="l1", max_iter=1000, solver="liblinear"),
        param_grid={"C": np.linspace(0.01, 0.1, 10)},
        cv=10,
        scoring="roc_auc",
    )
    clf.fit(X_train, y_train)

    print(f"Fitting LASSO with C={clf.best_params_['C']}")
    model = LogisticRegression(
        penalty="l1",
        max_iter=1000,
        solver="liblinear",
        C=clf.best_params_["C"],
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    model_predictions = model.predict_proba(X_test)[:, 1]

    # save model
    with open(
        os.path.join(config.data.model_path, f"{mode}_LASSO_{label_type}.pkl"), "wb"
    ) as f:
        pickle.dump(model, f)

    # Save predictions
    np.savez(
        os.path.join(
            config.data.save_predictions,
            f"{mode}_model_predictions_{label_type}.npz",
        ),
        model_predictions,
    )


def main(random_state: int = 42) -> None:
    """Main function which trains the model
    Args:
        random_state (int, 42): random state for reproducibility
    Returns:
        None
    """

    X_train_tabular, X_test_tabular, y_train, _ = preprocessing(
        feature_path=config.data.data_path,
        label_path=config.data.label_path,
        train_ids_path=config.data.train_ids,
        test_ids_path=config.data.test_ids,
        outcome=config.data.label_type,
        mode="tabular",
    )
    tabular_args = {
        "X_train": X_train_tabular,
        "y_train": y_train,
        "X_test": X_test_tabular,
        "mode": "tabular",
    }

    X_train_language, X_test_language, y_train, _ = preprocessing(
        feature_path=config.data.data_path,
        label_path=config.data.label_path,
        train_ids_path=config.data.train_ids,
        test_ids_path=config.data.test_ids,
        outcome=config.data.label_type,
        mode="language",
    )
    language_args = {
        "X_train": X_train_language,
        "y_train": y_train,
        "X_test": X_test_language,
        "mode": "language",
    }

    X_train_fusion, X_test_fusion, y_train, _ = preprocessing(
        feature_path=config.data.data_path,
        label_path=config.data.label_path,
        train_ids_path=config.data.train_ids,
        test_ids_path=config.data.test_ids,
        outcome=config.data.label_type,
        mode="fusion",
    )
    fusion_args = {
        "X_train": X_train_fusion,
        "y_train": y_train,
        "X_test": X_test_fusion,
        "mode": "fusion",
    }

    # Fit over the various Label types:
    for label_type in config.data.label_type:

        # ----------------------- Tabular Model ----------------------------
        if config.model.tabular_LASSO:
            # Tabular Data

            cv_fit_predict_LASSO(
                **tabular_args, label_type=label_type, random_state=random_state
            )

        # ----------------------- Language Model ----------------------------
        if config.model.language_LASSO:
            # TF-IDF Vectors

            cv_fit_predict_LASSO(
                **language_args, label_type=label_type, random_state=random_state
            )

        # ----------------------- Fusion Model ----------------------------
        if config.model.fusion_LASSO:
            # Tabular and TF-IDF vectors

            cv_fit_predict_LASSO(
                **fusion_args, label_type=label_type, random_state=random_state
            )


if __name__ == "__main__":
    main(random_state=config.seed)
