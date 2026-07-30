from flask import Flask,render_template,request
import numpy as np
import pandas as pd 
from scripts.prediction import make_predictions

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predictprice',methods=['GET','POST'])
def predict_price():
    if request.method=='GET':
     return render_template('home.html')
    else:
       bhk=int(request.form.get('bhk'))
       house_type=request.form.get('type')
       area_in_sq_ft=float(request.form.get('area'))
       region=request.form.get('region')
       status=request.form.get('status')
       age=request.form.get('age')

       input=pd.DataFrame(
          {
             'bhk':[bhk],
             'type':[house_type],
             'area':[area_in_sq_ft],
             'region':[region],
             'status':[status],
             'age':[age],
          }
       )
       pred=make_predictions(input)
       price_per_sq_ft=int((pred/area_in_sq_ft)*100000)
       return render_template('home.html',per_sq_ft=price_per_sq_ft,price=f'{pred:,.3f}')
       
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True,)