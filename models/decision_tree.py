from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

def create_model(preprocessor):
    
    # set max_depth = 4 for avoiding overfitting
    model = DecisionTreeClassifier(random_state=42, max_depth=4)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    return pipeline