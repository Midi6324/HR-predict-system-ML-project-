import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold

def evaluate_model_detailed(model, X_test, y_test, model_name):
    # show detailed evaluation report for a given model
    print(f"\n========================================")
    print(f"Evaluating the model: {model_name}")
    print(f"========================================")
    
    # 1. predict test set
    y_pred = model.predict(X_test)
    
    # 2. print classification report(accuracy, precision, recall, f1-score)
    print("--> Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"--> Confusion Matrix of {model_name}:")
    print(f"TP(VP): {cm[1][1]} | FP(FN): {cm[0][1]}")
    print(f"FN(FP): {cm[1][0]} | TN(VN): {cm[0][0]}")
    # 3.1 draw confusion matrix heatmap
    plot_confusion_matrix(model_name, cm)
    
    # 4. calculate metrics: f1-score, accuracy, AUC
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # trying calculate AUC by using whole pipeline
    # model perceptron and svm may not have predict_proba directly, so we need to handle it carefully
    auc = "N/A"
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            # some models return shape (n_samples, n_classes), some return (n_samples,)
            if proba.ndim == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(y_test, proba)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # decision_function may return shape (n_samples, n_classes) for some models
            if getattr(scores, "ndim", 1) == 2:
                auc = roc_auc_score(y_test, scores[:, 1])
            else:
                auc = roc_auc_score(y_test, scores)
    except Exception:
        pass

    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "F1-Score": f1,
        "AUC": auc
    }

def show_comparison_table(results_list):
    # show comparison table for all models

    results_df = pd.DataFrame(results_list)
    
    print("\n\n##########################################")
    print("MODEL ARENA - Comparison Summary")
    print("##########################################")
    
    # sort by F1-Score descending
    results_df = results_df.sort_values(by="F1-Score", ascending=False)
    
    # print table
    print(results_df.to_string(index=False))
    
    # send back the best model name
    best_model_name = results_df.iloc[0]["Model"]
    return best_model_name



def plot_confusion_matrix(model_name, cm):
    # draw confusion matrix heatmap
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_roc_curves(trained_models, X_test, y_test):
    # draw ROC curves competition for all models
    plt.figure(figsize=(8, 6))
    for name, pipeline in trained_models.items():
        try:
            # try probability first
            if hasattr(pipeline, 'predict_proba'):
                proba = pipeline.predict_proba(X_test)
                scores = proba[:, 1] if proba.ndim == 2 else proba
            # for models without predict_proba, try decision_function(svm, perceptron)
            elif hasattr(pipeline, 'decision_function'):
                scores = pipeline.decision_function(X_test)
                if getattr(scores, 'ndim', 1) == 2:
                    scores = scores[:, 1]
            else:
                continue

            fpr, tpr, _ = roc_curve(y_test, scores)
            plt.plot(fpr, tpr, label=name)
        except Exception:
            # skip models that fail to plot ROC
            continue

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('ROC Curves of All Models')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_model_roc(name, pipeline, X_test, y_test):
    # draw ROC curve for each single model
    try:
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(X_test)
            scores = proba[:, 1] if proba.ndim == 2 else proba
        elif hasattr(pipeline, 'decision_function'):
            scores = pipeline.decision_function(X_test)
            scores = scores[:, 1] if getattr(scores, 'ndim', 1) == 2 else scores
        else:
            print(f"Model {name} does not support probability scores for ROC curve.")
            return

        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.figure(figsize=(5.5, 4.5))
        plt.plot(fpr, tpr, label=f'{name}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.title(f'ROC Curve - {name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        print(f"Failed {name} drawing ROC curve.")

def plot_pr_curves(trained_models, X_test, y_test):
    # draw Precision-Recall curves competition for all models and show average precision(AP)
    plt.figure(figsize=(8, 6))
    for name, pipeline in trained_models.items():
        try:
            if hasattr(pipeline, 'predict_proba'):
                proba = pipeline.predict_proba(X_test)
                scores = proba[:, 1] if proba.ndim == 2 else proba
            elif hasattr(pipeline, 'decision_function'):
                scores = pipeline.decision_function(X_test)
                scores = scores[:, 1] if getattr(scores, 'ndim', 1) == 2 else scores
            else:
                continue
            precision, recall, _ = precision_recall_curve(y_test, scores)
            ap = average_precision_score(y_test, scores)
            plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
        except Exception:
            continue
    plt.title('Precision-Recall Curves of All Models')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.show()

def cross_validate_models(competitors, X, y, n_splits=5):
    # perform stratified 5-fold cross-validation for all models, return mean scores
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_summary = []
    for name, pipeline in competitors.items():
        auc_scores, f1_scores, acc_scores, ap_scores = [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_te)
            acc_scores.append(accuracy_score(y_te, y_pred))
            f1_scores.append(f1_score(y_te, y_pred))
            # calculate AUC and AP
            scores = None
            if hasattr(pipeline, 'predict_proba'):
                proba = pipeline.predict_proba(X_te)
                scores = proba[:, 1] if proba.ndim == 2 else proba
            elif hasattr(pipeline, 'decision_function'):
                scores = pipeline.decision_function(X_te)
                scores = scores[:, 1] if getattr(scores, 'ndim', 1) == 2 else scores
            if scores is not None:
                auc_scores.append(roc_auc_score(y_te, scores))
                ap_scores.append(average_precision_score(y_te, scores))
        cv_summary.append({
            'Model': name,
            'Acc(mean)': sum(acc_scores)/len(acc_scores) if acc_scores else None,
            'F1(mean)': sum(f1_scores)/len(f1_scores) if f1_scores else None,
            'AUC(mean)': sum(auc_scores)/len(auc_scores) if auc_scores else None,
            'AP(mean)': sum(ap_scores)/len(ap_scores) if ap_scores else None,
        })
    df = pd.DataFrame(cv_summary)
    print("\n\n===== 5-Fold Stratified Cross Validation (mean scores) =====")
    print(df.to_string(index=False))
    return df