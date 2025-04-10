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
        temperature=int(request.form.get("temperature"))
        Relative_Humidity = float(request.form.get("Relative Humidity"))
        mean_sea_level_pressure= float(request.form.get("mean sea level pressure"))
        total_precipitation = float(request.form.get(" total precipitation"))
        snowfall_amount= float(request.form.get("snowfall amount"))
        total_cloud_cover = float(request.form.get(" total cloud cover"))
        high_cloud_cover_high_cld_lay = float(request.form.get("high cloud cover high cld lay"))
        medium_cloud_cover_mid_cld_lay = float(request.form.get("medium cloud cover mid cld lay"))
        low_cloud = float(request.form.get("low cloud cover low cld lay"))
        shortwave_radiation_backwards = float(request.form.get("shortwave radiation backwards"))
        wind_speed_10m= float(request.form.get(" wind speed 10 m above gnd "))
        wind_direction_10m = float(request.form.get("wind direction 10 m above gnd "))
        wind_speed_80m = float(request.form.get("  wind speed 80 m abovr gnd"))
        wind_direction_80m = float(request.form.get(" wind direction 80 m above gnd "))
        wind_speed_900mb = float(request.form.get(" wind_speed_900_mb"))
        wind_direction_900mb = float(request.form.get(" wind_direction 900_mb"))
        wind_gust_10m_above_gnd = float(request.form.get("  wind_gust_10m_above_gnd ")) 
        angle_of_incidence = float(request.form.get(" angle_of_incidence ")) 
        zenith = float(request.form.get("  zenith ")) 
        azimuth = float(request.form.get(" azimuth"))
        generated_power_kW = float(request.form.get(" generation_power_kw ")) 


        new_data=scaler.transform([[temperature, Relative_Humidity,mean_sea_level_pressure, total_precipitation,snowfall_amount,total_cloud_cover,high_cloud_cover_high_cld_lay,medium_cloud_cover_mid_cld_lay,low_cloud ,shortwave_radiation_backwards,wind_speed_10m, wind_direction_10m, wind_speed_80m, wind_speed_900mb, wind_direction_900mb, wind_gust_10m_above_gnd,angle_of_incidence,zenith,azimuth,generated_power_kW]])
        result=model.predict(new_data)
            
         return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
