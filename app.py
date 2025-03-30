from flask import Flask,render_template,request
import numpy as np
import pandas as pd 
# from scripts.preprocessor import preprocess_data
# from scripts.model_trainer import train_model

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)