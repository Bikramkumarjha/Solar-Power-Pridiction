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
        generated_power_kw=int(request.form.get('generated_power_kw'))
        relative_humidity_2_m_above_gnd=float(request.form.get('relative_humidity_2_m_above_gnd'))
        mean_sea_level_pressure_MSL= float(request.form.get("mean_sea_level_pressure_MSL"))
        total_precipitation_sfc = float(request.form.get("total_precipitation_sfc"))
        snowfall_amount_sfc= float(request.form.get("snowfall_amount_sfc"))
        total_cloud_cover_sfc = float(request.form.get("total_cloud_cover_sfc"))
        high_cloud_cover_high_cld_lay= float(request.form.get("high_cloud_cover_high_cld_lay"))
        medium_cloud_cover_mid_cld_lay= float(request.form.get("medium_cloud_cover_mid_cld_lay"))
        low_cloud_cover_low_cld_lay= float(request.form.get("low_cloud_cover_low_cld_lay"))
        shortwave_radiation_backwards_sfc = float(request.form.get("shortwave_radiation_backwards_sfc"))
        wind_speed_10_m_above_gnd = float(request.form.get("wind_speed_10_m_above_gnd"))
        wind_direction_10_m_above_gnd == float(request.form.get("wind_direction_10_m_above_gnd"))
        wind_speed_80_m_above_gnd= float(request.form.get("wind_speed_80_m_above_gnd"))
        wind_direction_80_m_above_gnd= float(request.form.get("wind_direction_80_m_above_gnd"))
        wind_speed_900_mb = float(request.form.get("wind_speed_900_mb"))
        wind_direction_900_mb = float(request.form.get("wind_direction_900_mb"))
        wind_gust_10_m_above_gnd= float(request.form.get("wind_gust_10_m_above_gnd")) 
        angle_of_incidence= float(request.form.get("angle_of_incidence")) 
        zenith= float(request.form.get("zenith")) 
        azimuth = float(request.form.get("azimuth"))
        generated_power_kw = float(request.form.get("generated_power_kw"))
        

        new_data=scaler.transform([['temperature_2_m_above_gnd','relative_humidity_2_m_above_gnd',
       'mean_sea_level_pressure_MSL','total_precipitation_sfc',
       'snowfall_amount_sfc','total_cloud_cover_sfc',
       'high_cloud_cover_high_cld_lay','medium_cloud_cover_mid_cld_lay',
       'low_cloud_cover_low_cld_lay','shortwave_radiation_backwards_sfc',
       'wind_speed_10_m_above_gnd','wind_direction_10_m_above_gnd',
       'wind_speed_80_m_above_gnd','wind_direction_80_m_above_gnd',
       'wind_speed_900_mb','wind_direction_900_mb',
       'wind_gust_10_m_above_gnd','angle_of_incidence','zenith','azimuth',
       'generated_power_kw']])
        result=model.predict(new_data)
            
        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
