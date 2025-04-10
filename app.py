import pickle
from flask import Flask, request,jsonify,app,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application= Flask(__name__)
app=application

scaler=pickle.load(open("models/solar_power_scaler.pkl", "rb"))
model = pickle.load(open("models/solar_power_model.pkl", "rb"))

## Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temp=int(request.form.get("temp"))
        Rh=float(request.form.get("Rh"))
        mslp= float(request.form.get("mslp"))
        tp = float(request.form.get("tp"))
        sa= float(request.form.get("sa"))
        tcc = float(request.form.get("tcc"))
        hcc= float(request.form.get("hcc"))
        mcc = float(request.form.get("mcc"))
        lc= float(request.form.get("lc"))
        srb = float(request.form.get("srb"))
        wd10 = float(request.form.get("wd10"))
        ws80 = float(request.form.get("ws80"))
        wd80= float(request.form.get("wd80"))
        ws900 = float(request.form.get("ws900"))
        wd900 = float(request.form.get("wd900"))
        wg = float(request.form.get("wg")) 
        aoi = float(request.form.get("aoi")) 
        zenith = float(request.form.get("zenith")) 
        azimuth = float(request.form.get("azimuth"))

        new_data=scaler.transform([[temp,Rh,mslp,tp,sa,tcc,hcc,mcc,lc,srb,wd10,ws80,wd80,ws900,wd900,wg,aoi,zenith,azimuth]])
        result=model.predict(new_data)
            
        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
