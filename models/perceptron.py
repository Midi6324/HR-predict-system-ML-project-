from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# create a perceptron model
def create_model(preprocessor):
    # Wrap Perceptron to provide predict_proba for AUC
    base = Perceptron(random_state=42, max_iter=1000)
    model = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=5)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    return pipeline