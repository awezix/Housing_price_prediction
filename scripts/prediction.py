import pickle
import numpy as np
import pandas as pd

def make_predictions(input_data):
    # loading the models
    with open('models/preprocessor.pkl','rb') as f:
        preprocessor=pickle.load(f)

    with open('models/model.pkl','rb') as f:
        model=pickle.load(f)

    input_data_transformed=preprocessor.transform(input_data)
    predictions=model.predict(input_data_transformed)

    return predictions

if __name__ =="__main__":
    input=pd.DataFrame(
        {
            'bhk':[2],
            'type':['Apartment'],
            'area':[750],
            'region':['Vasai'],
            'status':['Under Construction'],
            'age':['New'],
        }
    )
    predict=make_predictions(input)
    print('predictions',predict)
