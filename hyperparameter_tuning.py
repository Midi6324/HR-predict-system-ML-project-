"""
Hyperparameter Tuning Module

Performs grid search on all 6 models using balanced parameter grids.

Logic:
1. Define parameter grids for each model (moderate number of combinations)
2. Use GridSearchCV with 5-fold cross-validation to search all combinations
3. For each combination: train model on 4 folds, validate on 1 fold, repeat 5 times
4. Compute average validation score (F1) across all 5 folds for each combination
5. Select the combination with highest average F1 score as best parameters
6. Return the best model (trained on full training set with best params)

This avoids overfitting by using CV to estimate generalization performance.
"""

from sklearn.model_selection import GridSearchCV
import pandas as pd


def get_param_grids():
    """
    Return moderate-sized parameter grids (balance speed vs accuracy)
    
    Each grid contains 6-36 combinations per model:
    - Too few: may miss optimal parameters
    - Too many: computationally expensive
    - Moderate: good trade-off for quick tuning
    """
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1, 10],
            'classifier__max_iter': [1000, 5000]
        },
        'Decision Tree': {
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        },
        'Random Forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 15, 20],
            'classifier__min_samples_split': [2, 5]
        },
        'Gaussian Naive Bayes': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        'Linear SVM': {
            'classifier__estimator__C': [0.1, 1, 10],
            'classifier__estimator__max_iter': [1000, 5000]
        },
        'Perceptron': {
            'classifier__estimator__alpha': [0.001, 0.01, 0.1],
            'classifier__estimator__max_iter': [1000, 5000]
        }
    }
    return param_grids


def tune_model(model_pipeline, X_train, y_train, model_name, cv=5, scoring='f1'):
    """
    Perform grid search on a single model
    
    Args:
        model_pipeline: sklearn Pipeline object (preprocessor + classifier)
        X_train: training feature set
        y_train: training label set
        model_name: model name (to lookup parameter grid)
        cv: number of cross-validation folds
        scoring: metric to optimize (default: f1-score)
    
    Returns:
        best_pipeline: model with best parameters (fitted on full training set)
        best_params: dictionary of best hyperparameters
        best_score: best cross-validation F1 score
    """
    param_grids = get_param_grids()
    
    if model_name not in param_grids:
        print(f"[Warning] No param grid defined for {model_name}, skipping tuning")
        return model_pipeline, {}, None
    
    param_grid = param_grids[model_name]
    
    print(f"\n[Tuning] Optimizing model: {model_name}")
    print(f"  Total combinations: {_count_combinations(param_grid)}")
    
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=1,  # single process to avoid Python 3.13 multiprocessing bug on macOS
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV score ({scoring}): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_all_models(competitors, X_train, y_train, cv=5, scoring='f1'):
    """
    Batch hyperparameter tuning for all models
    
    Args:
        competitors: dict, {model_name: pipeline}
        X_train: training feature set
        y_train: training label set
        cv: number of cross-validation folds
        scoring: metric to optimize
    
    Returns:
        tuned_competitors: dict, {model_name: best_pipeline}
        tuning_results: dict, {model_name: (best_params, best_score)}
    """
    tuned_competitors = {}
    tuning_results = {}
    
    print("\n" + "="*60)
    print("🔧 Hyperparameter Tuning Started")
    print("="*60)
    
    for model_name, pipeline in competitors.items():
        best_pipeline, best_params, best_score = tune_model(
            pipeline, X_train, y_train, model_name, cv=cv, scoring=scoring
        )
        tuned_competitors[model_name] = best_pipeline
        tuning_results[model_name] = (best_params, best_score)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning Completed")
    print("="*60)
    
    return tuned_competitors, tuning_results


def print_tuning_summary(tuning_results):
    """
    Print summary table of hyperparameter tuning results
    """
    print("\n" + "#"*60)
    print("📊 Tuning Summary")
    print("#"*60)
    
    summary_data = []
    for model_name, (best_params, best_score) in tuning_results.items():
        summary_data.append({
            'Model': model_name,
            'Best CV Score (F1)': f"{best_score:.4f}" if best_score else "N/A",
            'Num Params Tuned': len(best_params) if best_params else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


def _count_combinations(param_grid):
    """
    Calculate total number of parameter combinations
    """
    import numpy as np
    counts = [len(v) if isinstance(v, list) else 1 for v in param_grid.values()]
    return int(np.prod(counts))
