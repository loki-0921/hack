import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # Importing Support Vector Regression

# Fetch the latest COVID-19 data for USA
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields from the response
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
st.write(df)  # Display the latest data

# Bar chart for COVID-19 statistics
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
st.pyplot()  # Use Streamlit's function to display the plot

# Generate random historical data for cases and deaths (simulated)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Random data for the last 30 days
historical_deaths = np.random.randint(500, 2000, size=30)

# Create DataFrame for historical data
df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Display the historical data
st.write("Historical Data (Last 30 days):")
st.write(df_historical.head())

# Preparing data for SVM regression
X = df_historical[["day"]]  # Features (days)
y = df_historical["cases"]  # Target variable (cases)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM model for regression
svm_model = SVR(kernel='rbf', C=100, epsilon=0.1)  # 'rbf' kernel is a common choice

# Train the SVM model
svm_model.fit(X_train, y_train)

# Predict the cases for the next day (Day 31)
next_day = np.array([[31]])
predicted_cases = svm_model.predict(next_day)
st.write(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

# Streamlit Interface for prediction
st.title("COVID-19 Cases Prediction in USA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User input for prediction
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

# Button to trigger prediction
if st.button("Predict"):
    prediction = svm_model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
