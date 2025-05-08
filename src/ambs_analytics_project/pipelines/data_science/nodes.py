from typing import Tuple, Dict
import pandas as pd
import logging
from typing import List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier

logger = logging.getLogger(__name__)


def scale_numerical_columns(df: pd.DataFrame):
    """
    Separates numerical and binary columns in a DataFrame, and applies standard scaling to non-binary numerical columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with scaled numerical columns (excluding binary columns).
        list: List of scaled numerical column names.
        list: List of binary column names.
    """
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

    # Identify binary columns (with only 0 and 1 values)
    binary_columns = [
        col for col in numerical_columns if set(df[col].unique()).issubset({0, 1})
    ]

    # Non-binary numerical columns
    non_binary_numerical_columns = [
        col for col in numerical_columns if col not in binary_columns
    ]

    # Initialize the scaler
    scaler = StandardScaler()

    # Scale the non-binary numerical columns
    df[non_binary_numerical_columns] = scaler.fit_transform(
        df[non_binary_numerical_columns]
    )

    return df


def train_test_split_function(
    df_final: pd.DataFrame,
    target_col: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.

    Args:
        df_final (pd.DataFrame): The input DataFrame containing features and target.
        target_col (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - X_train: Training features
            - X_test: Testing features
            - y_train: Training target
            - y_test: Testing target
    """

    X = df_final.drop(columns=target_col)  # Features
    y = df_final[target_col]  # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def apply_sampling_smote_rus(
    X: pd.DataFrame, y: pd.Series, rus_params: dict, smote_params: dict
) -> tuple:
    """
    Apply Random Under Sampling (RUS) and SMOTE to balance the dataset.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        rus_params (dict): Parameters for Random Under Sampling.
        smote_params (dict): Parameters for SMOTE.

    Returns:
        tuple: Resampled features and target variable.
    """
    # Apply Random Under Sampling
    rus = RandomUnderSampler(**rus_params)
    X_rus, y_rus = rus.fit_resample(X, y)

    # Apply SMOTE
    smote = SMOTE(**smote_params)
    X_smote, y_smote = smote.fit_resample(X, y)

    return X_rus, y_rus, X_smote, y_smote


def feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    solver: str = "liblinear",
    class_weight: str = "balanced",
    max_iter: int = 10000,
    direction: str = "forward",
    scoring: str = "roc_auc",
    cv: int = 5,
    n_jobs: float = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature selection using Logistic Regression and retain only selected features.

    Parameters:
        X_train_scaled_df (pd.DataFrame): Scaled training data.
        y_train (pd.Series): Training labels.
        X_test_scaled_df (pd.DataFrame): Scaled testing data.
        solver (str): Solver to use in Logistic Regression.
        class_weight (str or dict): Class weight to use in Logistic Regression.
        max_iter (int): Maximum number of iterations for Logistic Regression.
        direction (str): Direction of feature selection ('forward' or 'backward').
        scoring (str): Scoring metric for feature selection.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        pd.DataFrame: Updated X_train with selected features.
        pd.DataFrame: Updated X_test with selected features.
        list: List of selected feature names.
    """

    # Ensure y_train is a 1D array
    y_train = y_train.values.ravel()

    # Logistic Regression model
    logreg = LogisticRegression(
        solver=solver, class_weight=class_weight, max_iter=max_iter
    )

    # Forward Feature Selection
    sfs = SequentialFeatureSelector(
        logreg, direction=direction, scoring=scoring, cv=cv, n_jobs=n_jobs
    )

    # Fit the selector
    sfs.fit(X_train, y_train)

    # Get selected feature names
    selected_features = X_train.columns[sfs.get_support()]
    logger.info(f"Selected features for all models: {list(selected_features)}")

    # Keep only selected features in scaled datasets
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    return X_train, X_test


def logistic_regression_node(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_rus: pd.DataFrame,
    y_train_rus: pd.Series,
    X_train_smote: pd.DataFrame,
    y_train_smote: pd.Series,
    params: Dict,
) -> LogisticRegression:
    """
    Tune a logistic regression model using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        LogisticRegression: Best logistic regression model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    max_iter = params.get("max_iter", 10000)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Ensure y_train is a 1D array
    y_train = y_train.values.ravel()
    y_train_rus = y_train_rus.values.ravel()
    y_train_smote = y_train_smote.values.ravel()

    # Step 1: Initialize logistic regression
    logreg_base = LogisticRegression(max_iter=max_iter, random_state=random_state)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=logreg_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    grid_search.fit(X_train_rus, y_train_rus)

    model_rus = grid_search.best_estimator_

    grid_search.fit(X_train_smote, y_train_smote)

    model_smote = grid_search.best_estimator_

    return model, model_rus, model_smote


def decision_tree_classifier_node(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_rus: pd.DataFrame,
    y_train_rus: pd.Series,
    X_train_smote: pd.DataFrame,
    y_train_smote: pd.Series,
    params: Dict,
) -> DecisionTreeClassifier:
    """
    Tune a Decision Tree Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        DecisionTreeClassifier: Best decision tree model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Ensure y_train is a 1D array
    y_train = y_train.values.ravel()
    y_train_rus = y_train_rus.values.ravel()
    y_train_smote = y_train_smote.values.ravel()

    # Step 1: Initialize decision tree classifier
    dt_base = DecisionTreeClassifier(random_state=random_state)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=dt_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best decision tree model
    model = grid_search.best_estimator_

    grid_search.fit(X_train_rus, y_train_rus)

    model_rus = grid_search.best_estimator_

    grid_search.fit(X_train_smote, y_train_smote)

    model_smote = grid_search.best_estimator_

    return model, model_rus, model_smote


def random_forest_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> RandomForestClassifier:
    """
    Tune a Random Forest Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        RandomForestClassifier: Best random forest model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Ensure y_train is a 1D array
    y_train = y_train.values.ravel()

    # Step 1: Initialize Random Forest Classifier
    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best Random Forest model
    model = grid_search.best_estimator_

    return model


def xgboost_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> XGBClassifier:
    """
    Tune an XGBoost Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        XGBClassifier: Best XGBoost model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Ensure y_train is a 1D array
    y_train = y_train.values.ravel()

    # Step 1: Initialize XGBoost Classifier with imbalance handling
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(
        y_train
    )  # Balance class weights

    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best XGBoost model
    model = grid_search.best_estimator_

    return model


def neural_network_classifier_node(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_rus: pd.DataFrame,
    y_train_rus: pd.Series,
    X_train_smote: pd.DataFrame,
    y_train_smote: pd.Series,
    params: Dict,
) -> MLPClassifier:
    """
    Tune an MLPClassifier (Neural Network) using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        MLPClassifier: Best MLP model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 10)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)
    max_iter = params.get("max_iter", 1000)

    # Ensure y_train is a 1D array
    y_train = y_train.values.ravel()
    y_train_rus = y_train_rus.values.ravel()
    y_train_smote = y_train_smote.values.ravel()

    # Step 1: Initialize MLP Classifier
    mlp_base = MLPClassifier(max_iter=max_iter, random_state=random_state)

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=mlp_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    grid_search.fit(X_train_rus, y_train_rus)

    model_rus = grid_search.best_estimator_

    grid_search.fit(X_train_smote, y_train_smote)

    model_smote = grid_search.best_estimator_

    return model, model_rus, model_smote


def catboost_classifier_node(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> CatBoostClassifier:
    """
    Tune a CatBoost Classifier using GridSearchCV with configurable parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (Dict): Dictionary of parameters including 'param_grid' and other configurations.

    Returns:
        CatBoostClassifier: Best CatBoost model after hyperparameter tuning.
    """
    # Extract parameters
    param_grid = params["param_grid"]
    scoring = params.get("scoring", "recall")
    cv = params.get("cv", 5)
    random_state = params.get("random_state", 42)
    verbose = params.get("verbose", 1)
    n_jobs = params.get("n_jobs", -1)

    # Ensure y_train is a 1D array
    y_train = y_train.values.ravel()

    # Step 1: Initialize CatBoost Classifier
    cb_base = CatBoostClassifier(
        verbose=0, random_state=random_state  # Suppress CatBoost training logs
    )

    # Step 2: Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=cb_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    grid_search.fit(X_train, y_train)

    # Best CatBoost model
    model = grid_search.best_estimator_

    return model


def stacked_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> StackingClassifier:
    """
    Train a stacked classifier with specified base learners and meta-learner.

    Args:
        data (pd.DataFrame): The input dataset.
        target_column (str): Name of the target column.
        test_size (float): Proportion of data to use as test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Trained model and evaluation metrics.
    """

    y_train = y_train.values.ravel()

    # Define base learners
    base_learners = [
        (
            "lr",
            LogisticRegression(
                C=0.01,
                class_weight="balanced",
                penalty="l2",
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                class_weight="balanced",
                max_depth=10,
                max_features="sqrt",
                min_samples_leaf=5,
                min_samples_split=2,
                n_estimators=1500,
                random_state=42,
            ),
        ),
        (
            "dt",
            DecisionTreeClassifier(
                class_weight="balanced",
                criterion="gini",
                max_depth=3,
                max_features="sqrt",
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=42,
            ),
        ),
        (
            "mlp",
            MLPClassifier(
                activation="relu",
                alpha=0.0001,
                hidden_layer_sizes=(128, 64, 32),
                learning_rate="constant",
                solver="adam",
                random_state=42,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                learning_rate=0.1,
                max_depth=4,
                min_child_weight=5,
                n_estimators=100,
                scale_pos_weight=10,
                subsample=1.0,
                random_state=42,
            ),
        ),
        (
            "catboost",
            CatBoostClassifier(
                auto_class_weights="Balanced",
                bootstrap_type="Bayesian",
                depth=4,
                iterations=300,
                l2_leaf_reg=5,
                learning_rate=0.1,
                random_state=42,
                verbose=0,
            ),
        ),
    ]

    # Define meta-learner
    meta_learner = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )

    # Stacking Classifier
    stack_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=True,
    )

    stack_model.fit(X_train, y_train)

    return stack_model


def results(
    stacked_model,
    catboost_model,
    neural_network_model_rus,
    xgboost_model,
    random_forest_model,
    decision_tree_model,
    logistic_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Evaluates pre-trained models stored in the catalog.

    Args:
        model_names
        catalog (dict): Kedro catalog containing the saved models.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): True labels for training set.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): True labels for testing set.
    """

    models = [
        stacked_model,
        catboost_model,
        neural_network_model_rus,
        xgboost_model,
        random_forest_model,
        decision_tree_model,
        logistic_model,
    ]

    results_list = []

    for model in models:
        logger.info(f"Evaluating model: {type(model).__name__}")

        # Predict probabilities
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Predict classes
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute AUC scores
        auc_train = roc_auc_score(y_train, y_train_proba)
        auc_test = roc_auc_score(y_test, y_test_proba)

        # Calculate Precision and Recall

        precision_train_1 = precision_score(y_train, y_train_pred, pos_label=1)
        recall_train_1 = recall_score(y_train, y_train_pred, pos_label=1)
        precision_train_0 = precision_score(y_train, y_train_pred, pos_label=0)
        recall_train_0 = recall_score(y_train, y_train_pred, pos_label=0)

        precision_test_1 = precision_score(y_test, y_test_pred, pos_label=1)
        recall_test_1 = recall_score(y_test, y_test_pred, pos_label=1)
        precision_test_0 = precision_score(y_test, y_test_pred, pos_label=0)
        recall_test_0 = recall_score(y_test, y_test_pred, pos_label=0)

        # Log AUC scores
        logger.info(f"TRAIN AUC: {auc_train:.4f}")
        logger.info(f"TEST AUC: {auc_test:.4f}")
        logger.info(f"precision_train_1: {precision_train_1:.4f}")
        logger.info(f"recall_train_1: {recall_train_1:.4f}")
        logger.info(f"precision_train_0: {precision_train_0:.4f}")
        logger.info(f"recall_train_0: {recall_train_0:.4f}")
        logger.info(f"precision_test_1: {precision_test_1:.4f}")
        logger.info(f"recall_test_1: {recall_test_1:.4f}")
        logger.info(f"precision_test_0: {precision_test_0:.4f}")
        logger.info(f"recall_test_0: {recall_test_0:.4f}")

        # Confusion Matrix
        TN, FP, FN, TP = confusion_matrix(y_test, y_test_pred).ravel()
        logger.info(f"True-Negatives (TN): {TN}")
        logger.info(f"False-Positives (FP): {FP}")
        logger.info(f"False-Negatives (FN): {FN}")
        logger.info(f"True-Positives (TP): {TP}")

        # Expected Profit Calculation
        exp_profit = TN - FN - FP
        logger.info(f"Expected Profit: {exp_profit}")

        results_list.append(
            {
                "Model": type(model).__name__,
                "Train AUC": auc_train,
                "Test AUC": auc_test,
                "precision_train_1": precision_train_1,
                "recall_train_1": recall_train_1,
                "precision_train_0": precision_train_0,
                "recall_train_0": recall_train_0,
                "precision_test_1": precision_test_1,
                "recall_test_1": recall_test_1,
                "precision_test_0": precision_test_0,
                "recall_test_0": recall_test_0,
                "True Negatives": TN,
                "False Positives": FP,
                "False Negatives": FN,
                "True Positives": TP,
                "Expected Profit": exp_profit,
            }
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df
