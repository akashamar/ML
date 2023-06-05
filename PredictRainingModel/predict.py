import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# read 23 years weather report 
df = pd.read_csv('2000_2023.csv')

# precipitation greater than or equal to 2.5mm means rain
df['rain'] = df['precipitation'].apply(lambda x: 1 if x >= 2.5 else 0)

# remove precipitation and time column
df = df.drop(['precipitation', 'time'], axis=1)

# Get the column names of the dataset
columns = df.columns

# check how does particular value effect the rain 
for col1 in columns:
    # Create a scatter plot
    plt.scatter(df[col1], df["rain"])
    plt.xlabel(col1)
    plt.ylabel('rain')
    plt.title(f'{col1} vs rain')
    # plt.show()

# Separate the features (X) and the target variable (y)
X = df.drop('rain', axis=1)  # Features (all columns except 'rain')
y = df['rain']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model using the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Convert predicted probabilities to binary labels based on a threshold of 0.5
# Values above or equal to 0.5 were classified as 1 (rain), and values below 0.5 were classified as 0 (non-rain)
y_pred_labels = (y_pred >= 0.5).astype(int)

# Calculate accuracy
accuracy = (y_pred_labels == y_test.values).mean()

print("Accuracy:", accuracy)

# read all the given values
temperature = float(input("Enter temperature : "))
shortwave_radiation = float(input("Enter shortwave radiation : "))
wind_speed =  float(input("Enter wind speed : "))
wind_gusts =  float(input("Enter wind gusts : "))
wind_direction =  float(input("Enter wind direction : "))
et0_fao_evapotranspiration =  float(input("Enter evapotranspiration : "))

# Create a new feature matrix with manual values
user_values = [
    [ temperature, shortwave_radiation, wind_speed, wind_gusts, wind_direction, et0_fao_evapotranspiration],
]

# Convert the manual values into a DataFrame
manual_df = pd.DataFrame(user_values, columns=X_train.columns)

# Make predictions on the manual values
manual_predictions = model.predict(manual_df)

if manual_predictions >= 0.5:
    prediction = "It will rain today"
else:
    prediction = "It will not rain today"

print(prediction)