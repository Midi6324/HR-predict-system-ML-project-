from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def create_model(preprocessor):
# create a random forest model with max_depth=4 to avoid overfitting
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=4)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    return pipeline
