from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# create a support vector machine model
def create_model(preprocessor):
    # Wrap LinearSVC to provide predict_proba for AUC
    base = LinearSVC(random_state=42, max_iter=1000)
    model = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=5)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    return pipeline