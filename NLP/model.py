from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

  
def build_model():
    """This function builds a new model and returns it.

    The model is implemented as a sklearn Pipeline object.

    The pipeline has two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of the model
    """

    preprocessor = ColumnTransformer(
        [
            ("processing", TfidfVectorizer(), "text")
        ]
    )
    model = SGDClassifier(alpha=0.0005)
    
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
