
# Predicting Driver Positions in Formula 1 Grand Prix Using Historical Race Data

## Background

Formula 1 (F1) is one of the most competitive motorsports, where race outcomes are influenced by a wide variety of factors, including driver skill, team performance, track characteristics, weather conditions, and in-race incidents. The ability to predict the finishing positions of drivers in a race can provide valuable insights for teams, analysts, and fans. This project aims to leverage historical race data to build a predictive model for F1 race outcomes.

## Objective

The objective of this project is to develop a **machine learning model** that can predict the finishing position of a driver in an upcoming Formula 1 race based on historical race data. By analyzing data from past races, we aim to identify key performance indicators that contribute to race success and position determination.

## Problem Type

This is a **regression problem** where the model predicts the **finishing position** of a driver in a given race. The output will be a continuous value representing the predicted position (e.g., 1st, 2nd, 3rd, etc.).

## Challenges

- **Feature Selection:** Choosing the right features that have the most significant impact on the driver's finishing position.
- **Unpredictability:** Various external factors like crashes, tire failures, or strategy changes that can lead to unexpected race outcomes.
- **Data Imbalance:** Some drivers may dominate certain seasons or races, making it harder to predict the positions of lower-ranked drivers.
- **Complexity:** Accounting for the interactions between weather conditions, race strategy, and driver skills.

## Evaluation Metrics

The following metrics were used to evaluate the performance of the model:

- **Mean Absolute Error (MAE):** Measures the average absolute error between the predicted and actual positions.
- **Root Mean Squared Error (RMSE):** Measures the square root of the average squared differences between predicted and actual positions, highlighting large errors.
- **R-Squared (R²):** Evaluates how well the model explains the variance in the driver's finishing positions.

## Expected Outcome

The goal of this project is to develop a machine learning model capable of predicting the likely finishing position of a driver in an upcoming Formula 1 race, based on historical race data. The model will offer insights into the key factors that influence race outcomes.

## Features of the Streamlit App

- **Input Parameters:**
  - **Basic Race Info:** Year, Round, Grand Prix Name, Country
  - **Race Inputs:** Driver, Team, Status, First Compound
  - **Race Conditions:** Grid Position, Points Scored, Mean Lap Time, Std Lap Time, Pit Stops, Stint Count, Air Temperature, Rainfall, Humidity, Wind Speed, Track Temperature
  
- **User-Friendly Interface:** All inputs are customizable through easy-to-use dropdowns, sliders, and number inputs.
- **Model Prediction:** Upon entering the race details, the app predicts the driver's finishing position in the upcoming race.
  
## Technologies Used

- **Python**: Programming language for implementing the machine learning model and app.
- **Streamlit**: Web framework to build and deploy the app.
- **XGBoost**: Machine learning algorithm for building the predictive model.
- **Pandas**: Data manipulation and analysis library for handling the race data.
- **Scikit-learn**: For evaluation metrics and preprocessing tasks.

## How to Run Locally

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/neelayjain02/f1-position-predictor.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501` to use the app.

## Deployment

This app is deployed on [Streamlit Cloud](https://f1-driver-position-predictor.streamlit.app/). You can access the live version of the app [here](https://f1-driver-position-predictor.streamlit.app/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
