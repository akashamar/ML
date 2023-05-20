import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ask userExperience from user
userExperience = 0
try:
  userExperience = int(input("What is your experience? "))
except ValueError:
  print("Invalid input. Please enter a number.")
else:
  print("Your experience", userExperience, "years")

# We will then need to read the dataset into a Pandas DataFrame:
df = pd.read_csv('salary_data.csv')

# The next step is to split the data into the independent and dependent variables:
x = df['Years of Experience'].values.reshape(-1, 1)
y = df['Salary'].values.reshape(-1, 1)

# We will then need to fit a linear regression model to the data:
model = LinearRegression()
model.fit(x, y)

# Once the model is fitted, we can use it to predict the salary of a person using experience:
prediction = model.predict([[userExperience]])

print("Your salary can be: ", prediction[0][0])