from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# create a logistic regression model
def create_model(preprocessor):
    
    # max_iter=1000 to ensure convergence
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    return pipeline