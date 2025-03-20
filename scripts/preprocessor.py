import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

def preprocess_data(df):
    df.columns = df.columns.str.strip().str.lower()

    numerical_features=["bhk","area"]
    categorical_feature=["type","region","status","age"]

    num_transformers=Pipeline(
        steps=[
            ("Imputer",SimpleImputer(strategy="mean")),
            ("Scaler",StandardScaler())
        ]
        )
    cat_transformers=Pipeline(
        steps=[
            ("Imputer",SimpleImputer(strategy="most_frequent")),
            ("Encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
        ]
    )

    preprocessor=ColumnTransformer(
        transformers=[
            ("numerical_transformer",num_transformers,numerical_features),
            ("categorical_transformer",cat_transformers,categorical_feature)
        ]
    )

    return preprocessor

if __name__=="__main__":
    # ex for testing above code
    df=pd.read_csv("Datasets/cleaned_data.csv")
    df_processed=preprocess_data(df)

    df_transformed=df_processed.fit_transform(df)
    print("preprocessing done")

    with open("models/preprocessor.pkl","wb") as f:
        pickle.dump(df_processed,f)
    print("preprocessor pipeleine saved ad pickle")