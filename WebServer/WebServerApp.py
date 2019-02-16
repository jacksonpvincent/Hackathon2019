import os
import json
import pandas as pd
from flask import Flask
from flask import Flask, request
from flask import request, jsonify
from ast import literal_eval
from pandas.core.frame import DataFrame

import sys
sys.path.insert(0, '../KerasFeasibility')

from DataFeeder import DataFeeder

df = pd.DataFrame(columns=['Id', 'Cycle' , 'LtLt', 'LtRt', 'RtLt', 'RtRt'])
dataFeeder = DataFeeder(df)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def result():
    # request.form to get form parameter
    if request.method == 'POST':
        dataSet = request.data
        data = literal_eval(dataSet.decode('utf8'))

        #jsonString = dataSet.decode('utf-8').replace("'", '"')
        #for dataField in dataSet:
        #    row = dataField;
        #s = json.dumps(data, indent=4, sort_keys=True)
        #print(s)
        
        dataDictionary = json.loads(json.dumps(data, indent=4, sort_keys=True))
        record = [(1, 1, dataDictionary[0]["value"] ,dataDictionary[1]["value"], dataDictionary[2]["value"], dataDictionary[3]["value"])]
        df = pd.DataFrame.from_records(record)
        dataFeeder.SetDataRecord(df)
        
        #df.append(pd.DataFrame.from_records(record))
        #print(df)
        #oneRowData ="1,1," + dataDictionary[0]["value"] + "," + dataDictionary[1]["value"] + "," + dataDictionary[2]["value"] + "," + dataDictionary[3]["value"]

           
        return "Received" 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')