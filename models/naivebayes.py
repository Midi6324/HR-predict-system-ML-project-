from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _to_dense(x):
    # Convert sparse matrix to dense; keep dense as it is.
    return x.toarray() if hasattr(x, "toarray") else x

def create_model(preprocessor):
    # Convert sparse matrix (OneHot) to dense so GaussianNB can consume it
    to_dense = FunctionTransformer(
        _to_dense,
        accept_sparse=True,
        feature_names_out="one-to-one",
    )

    model = GaussianNB()
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('to_dense', to_dense),
        ('classifier', model)
    ])
    return pipeline