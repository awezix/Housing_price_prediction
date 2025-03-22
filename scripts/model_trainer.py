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

def performance_metrics(y_true,y_pred):
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true,y_pred)
    return mae,mse,rmse,r2

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
    
    best_model={
        'name':None,
        'model':None,
        'score':None
    }

    for name,model in models.items():
        model.fit(x_train_transformed,y_train)
        y_test_pred=model.predict(x_test_transformed)

        mae,mse,rmse,r2=performance_metrics(y_test,y_test_pred)
        print(name)
        print("model performance on testing set")
        print("MAE:",mae)
        print("MSE:",mse)
        print("RMSE:",rmse)
        print("R2 Score:",r2)
        
        print("="*35)
        print("\n")

        if best_model['score'] is None or r2 > best_model['score']:
            best_model={
                'name':name,
                'model':model,
                'score':r2
            }
        
    if best_model['model']:
        with open("models/model.pkl",'wb') as f:
            pickle.dump(best_model["model"],f)
            print(f'saved the best model {best_model["name"]}')

if __name__ =="__main__":
    df=pd.read_csv("Datasets/cleaned_data.csv")
    train_model(df) 

