import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from preprocessor import preprocess_data
import pickle

def train_model(df):

    x=df.drop("price_in_lakh",axis=1)
    y=df["price_in_lakh"]

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

    with open("models/preprocessor.pkl","rb") as f:
        preprocessor=pickle.load(f)

    x_train_transformed=preprocessor.fit_transform(x_train)
    x_test_transformed=preprocessor.transform(x_test)
    
    models={
        "linear regression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "Decision tree":DecisionTreeRegressor(),
        "Random forest":RandomForestRegressor(),
        "Adaboost":AdaBoostRegressor(),
        "Xgboost":XGBRegressor()
    }
    
    for i in range(len(list(models))):
        model=list(models.values())[i]
        model.fit(x_train_transformed,y_train)
        y_test_pred=model.predict(x_test_transformed)
        
        print(list(models.keys())[i])
        print("model performance on testing set")
        print("MAE:",mean_absolute_error(y_test,y_test_pred))
        print("MSE:",mean_squared_error(y_test,y_test_pred))
        print("RMSE:",np.sqrt(mean_squared_error(y_test,y_test_pred)))
        print("R2 Score:",r2_score(y_test,y_test_pred))
        
        print("="*35)
        print("\n")

if __name__ =="__main__":
    df=pd.read_csv("Datasets/cleaned_data.csv")
    train_model(df) 


