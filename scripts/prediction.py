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
    return predictions[0]    #only 3 digit after decimal

if __name__ =="__main__":
    input=pd.DataFrame(
        {
            'bhk':[2],
            'type':['Apartment'],
            'area':[800],
            'region':['Thane West'],
            'status':['Ready to move'],
            'age':['Resale'],
        }
    )
    predict=make_predictions(input)
    print('predictions',predict)
