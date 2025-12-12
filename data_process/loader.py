# data loader: data_process/loader.py
import pandas as pd
from sklearn.model_selection import train_test_split # spliter for train and test
from sklearn.preprocessing import StandardScaler, OneHotEncoder # scaler and encoder for nonumeric data
from sklearn.impute import SimpleImputer # imputer for missing data
from sklearn.compose import ColumnTransformer # pipeline for different columns
from sklearn.pipeline import Pipeline # the great pipeline

def get_data_pipeline(csv_path):
    """
    creat the data loading and preprocessing pipeline:
    1. fill missing values with (median for numeric and 'missing' for categorical
    2. transform categorical data to nueric data with OneHotEncoder
    3. standardization
    """
    print("[Loader] get_data_pipeline called with:", csv_path)
    # 1.LOAD DATA
    try:
        df = pd.read_csv(csv_path)
        # normalize column names to improve blacklist matching
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Can't find the path: {csv_path}，check the file path.")

    # separate features X and target y
    # add a *****blacklist****** mechanism to avoid possible data leakage(if exist)
    # parameter in dataset contains:'database contains 'longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity'
    
    blacklist = [
        ''
    ]
    cols_to_drop = [c for c in blacklist if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
        # sanity log to ensure blacklist applied
        print(f"[Loader] Dropped leakage parameters: {cols_to_drop}")
    else:
        print(f"[Loader] No blacklist parameters found in CSV header: {list(df.drop('Sold6M', axis=1).columns)}")

    X = df.drop("Sold6M", axis=1)
    if any(c in X.columns for c in blacklist):
        raise RuntimeError("Blacklist parameters still present in features; check data loading path or column names.")
    y = df["Sold6M"]


    # 2. train-test dataset split(80%train, 20%test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. PREPROCESSING PIPELINE SETUP
    # numeric columns treatment(eg. median impute + standard scaler)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=
    [
        ('imputer', SimpleImputer(strategy='median')),  # replace missing values with it's median
        ('scaler', StandardScaler())                    # standardization
    ])

    # categorical columns treatment(eg. missing impute + one hot encoding)
    categorical_features = X.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # replace missing values with mode
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) 
        # one-hot encoding, ignore the unknown categories if appear
        # sparse_output=True makes conflict with GaussianNB in Naive Bayes model
        # introduce pipeline to avoid this issue in 'naivebayes.py'

    ])

    # combine both numeric and categorical transformers       
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X_train, X_test, y_train, y_test, preprocessor

