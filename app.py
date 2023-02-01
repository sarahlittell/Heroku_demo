import pandas as pd
from flask import Flask, request,jsonify
import pickle

app=Flask(__name__)
model=pickle.load(open('Car_Prices_model.pickle', 'rb'))

@app.route('/', methods =['GET', 'POST'])
def home():
    if(request.method=='GET'):
        data='hello world :)'
        return jsonify({'data':data})

@app.route('/predict/')
def predict():
    year=request.args.get('year')
    miles=request.args.get('miles')

    test_df=pd.DataFrame({'Year':[year],'Miles':[miles]})
    pred_price=model.predict(test_df)
    return jsonify({'Car Price':str(pred_price)})

    
if __name__=="__main__":
    app.run(debug=True)