rom sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


num_col = [column for column in X_train.columns if X_train[column].dtype != 'object']
cate_col = [column for column in X_train.columns if X_train[column].dtype = 'object']


def build_model():
    """This function builds a new model and returns it.

    The model is implemented as a sklearn Pipeline object.

    Your pipeline has two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of the model
    """

    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ("num_cols", "passthrough")])

    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_col),
        ('cat', categorical_transformer, cate_col)])

    model = XGBClassifier(
        max_depth=7,
        min_child_weight=5,
        n_estimators=50,
        colsample_bytree=0.8,
        subsample=0.8
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])
