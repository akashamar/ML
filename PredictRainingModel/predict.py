import numpy as np
import pandas as pd
import requests
import colorama
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def determine_rain(x):
    if x >= 2.5:
        return 1
    else:
        return 0

def read_weather_data_from_csv(filename):
    df = pd.read_csv(filename)
    df['rain'] = df['precipitation'].apply(determine_rain)
    df = df.drop(['precipitation', 'time'], axis=1)
    return df

def read_weather_data_from_api(X_train, api_url):
    response = requests.get(api_url)
    data = response.json()
    dailyData = data["daily"]
    temperature, shortwave_radiation, wind_speed, wind_gusts, wind_direction, et0_fao_evapotranspiration = dailyData["temperature_2m_max"][0], dailyData["shortwave_radiation_sum"][0], dailyData["windspeed_10m_max"][0], dailyData["windgusts_10m_max"][0], dailyData["winddirection_10m_dominant"][0], dailyData["et0_fao_evapotranspiration"][0]
    user_values = [
        [temperature, shortwave_radiation, wind_speed, wind_gusts, wind_direction, et0_fao_evapotranspiration],
    ]
    print("Temperature:", temperature)
    print("Shortwave Radiation:", shortwave_radiation)
    print("Wind Speed:", wind_speed)
    print("Wind Gusts:", wind_gusts)
    print("Wind Direction:", wind_direction)
    print("ET0 FAO Evapotranspiration:", et0_fao_evapotranspiration)
    manual_df = pd.DataFrame(user_values, columns=X_train.columns)
    return manual_df

def main():
    df = read_weather_data_from_csv('2000_2023.csv')
    X = df.drop('rain', axis=1)
    y = df['rain']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred >= 0.5).astype(int)
    accuracy = (y_pred_labels == y_test.values).mean()
    print("Model Accuracy:", accuracy)
    manual_df = read_weather_data_from_api(X_train, "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&daily=temperature_2m_max,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration&timezone=Asia%2FSingapore")
    manual_predictions = model.predict(manual_df)
    # Prediction part with color
    prediction = "It will rain today" if manual_predictions >= 0.5 else "It will not rain today"
    if manual_predictions >= 0.5:
        prediction_color = colorama.Fore.BLUE + prediction + colorama.Fore.RESET
    else:
        prediction_color = colorama.Fore.RED + prediction + colorama.Fore.RESET
    print(prediction_color)

if __name__ == "__main__":
    main()
