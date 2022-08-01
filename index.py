from flask import Flask, jsonify, request, make_response
from flask_cors import CORS , cross_origin
from flask import json
import numpy as np
import pandas as pd
import joblib
import json
import sklearn
import pickle

config = {
  'ORIGINS': [
    'http://localhost:8000',  # React
    'http://127.0.0.1:8000',  # React
  ],

  'SECRET_KEY': '...'
}


app= Flask(__name__)

CORS(app, resources={ r'/*': {'origins': config['ORIGINS']}}, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/resultanalysis",methods=["POST"])

def main():
    s = request.json
    ns=s['detailedreport']
    print(ns)
    data =ns
    
    ndata=[]
    n=[]
    for i in data:
      n.append(int(data[i]))
    ndata.append(n)
    xfresh=pd.DataFrame(np.array(ndata))
    print(ndata)


    knn = joblib.load("./knnmodel")
    result = knn.predict(xfresh)[0]


    print(result)


    response = make_response(
    jsonify(
        {"prediction":result}
    ),
    200,
    )
    response.headers["Content-Type"] = "application/json"
    return response

if __name__=="__main__":
    app.run(debug=True)