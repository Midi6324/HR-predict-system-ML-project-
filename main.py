import pandas as pd
from data_process.loader import get_data_pipeline
from evaluation.evaluator import evaluate_model_detailed, show_comparison_table, plot_roc_curves, plot_model_roc, plot_pr_curves, cross_validate_models
from hyperparameter_tuning import tune_all_models, print_tuning_summary
import models.logistic as logistic
import models.perceptron as perceptron
import models.decision_tree as decision_tree
import models.randomforest as randomforest
import models.svm as svm
import models.naivebayes as naivebayes

def main(enable_hyperparameter_tuning=False):
    print("Launching House Pricing 6-Month Prediction Model System...")
    
    # 1. prepare data
    print("\n[Step 1] Data loading...")
    X_train, X_test, y_train, y_test, preprocessor = get_data_pipeline("housing6month.csv")
    
    # 2. define models to compare
    competitors = {
        "Logistic Regression": logistic.create_model(preprocessor),
        "Perceptron":        perceptron.create_model(preprocessor),
        "Decision Tree":     decision_tree.create_model(preprocessor),
        "Random Forest":    randomforest.create_model(preprocessor),
        "Support Vector Machine": svm.create_model(preprocessor),
        "Naive Bayes":      naivebayes.create_model(preprocessor)
    }

    # 2.5 Optional: Hyperparameter tuning on training set
    if enable_hyperparameter_tuning:
        print("\n[Step 2.5] Hyperparameter Tuning...")
        competitors, tuning_results = tune_all_models(competitors, X_train, y_train, cv=5, scoring='f1')
        print_tuning_summary(tuning_results)
    else:
        print("\n[Step 2.5] Skipping Hyperparameter Tuning...")

    # 3. train and evaluate each model
    print("\n[Step 3] Starting model training and evaluation...")
    # trained_models in dictionary
    trained_models = {}
    # resalt in list for final comparison
    results_summary = [] 
    
    for name, pipeline in competitors.items():
        # A. Training process
        print(f"\nTraining{name} ...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        
        # B. Show detailed evaluation
        metrics = evaluate_model_detailed(pipeline, X_test, y_test, name)
        results_summary.append(metrics)

        # C. Plot individual ROC curve for this model
        plot_model_roc(name, pipeline, X_test, y_test)

    # 4. show comparison table
    print("\n[Step 4] Comparing...")
    best_name = show_comparison_table(results_summary)
    
    # draw all models' ROC curves
    plot_roc_curves(trained_models, X_test, y_test)
    # draw all models' Precision-Recall curves
    plot_pr_curves(trained_models, X_test, y_test)

    # 5. perform 5-fold cross-validation on training set
    print("\n[Step 5] 5-Fold Stratified Cross-Validation on Training Set...")
    cross_validate_models(competitors, X_train, y_train)
    
    print(f"\nThe best model is: {best_name}")


if __name__ == "__main__":
    # Set enable_hyperparameter_tuning to True to perform hyperparameter tuning
    main(enable_hyperparameter_tuning=True)