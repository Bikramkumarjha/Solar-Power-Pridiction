from flask import Flask, request,jsonify,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application

scaler=pickle.load(open("/solar_power_scaler.pkl", "rb"))
model = pickle.load(open("/solar_power_model.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint()
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
        azimuth = float(request.form.get(" azimuth"))
        generated_pow = float(request.form.get("generated_pow")) 


        new_data=scaler.transform([[temperature, Relative_Humidity,mean_sea_level_pressure, total_precipitation,snowfall_amount,total_cloud_cover,high_cloud_cover_high_cld_lay,medium_cloud_cover_mid_cld_lay,low_cloud ,shortwave_radiation_backwards,wind_direction_10m, wind_speed_80m,wind_direction_80m, wind_speed_900mb, wind_direction_900mb, wind_gust_10m_above_gnd,angle_of_incidence,zenith,azimuth,generated_power_kW]])
        result=model.predict(new_data)
            
        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
