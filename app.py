# Import necessary libraries at the beginning of your script
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet

app = Flask(__name__)

# Assuming 'y' is your time series data
# In a real application, you should replace this with your actual data loading logic

def get_views_info(user_date):
    data = pd.read_csv("train_2_views.csv")
    data = pd.melt(data, id_vars='Page', var_name='date', value_name='Visits')

    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['weekday'] = data['date'].apply(lambda x: x.weekday())

    # Filter the DataFrame for visits greater than 0
    valid_dates_df = data[data['Visits'] > 0]

    # Group by date and find the rows with the minimum and maximum visits for each date
    min_visits_row = valid_dates_df.loc[valid_dates_df.groupby('date')['Visits'].idxmin()]
    max_visits_row = valid_dates_df.loc[valid_dates_df.groupby('date')['Visits'].idxmax()]

    # Filter rows for the user-provided date
    min_visits_row = min_visits_row[min_visits_row['date'] == user_date]
    max_visits_row = max_visits_row[max_visits_row['date'] == user_date]

    if min_visits_row.empty or max_visits_row.empty:
        return None, None

    return min_visits_row.to_dict(orient='records')[0], max_visits_row.to_dict(orient='records')[0]


def plot_visits_by_month(df, page, year):
    # Extracting relevant columns based on the selected year
    selected_columns = [col for col in df.columns if str(year) in col]
    
    # Selecting the specific row for the given page
    selected_row = df[df['Page'] == page]
    
    # Extracting data for the selected page and year
    visits_data = selected_row[selected_columns].values.flatten()
    
    # Aggregate visits by month
    monthly_visits = pd.Series(visits_data).groupby(pd.to_datetime(selected_columns).to_period("M")).sum()
    
    # Plotting the bar graph
    plt.figure(figsize=(10, 6))
    monthly_visits.plot(kind='bar', rot=45)
    plt.title(f'Visits for {page} in {year} (Aggregated by Month)')
    plt.xlabel('Month')
    plt.ylabel('Number of Visits')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the plot as base64 string for embedding in HTML
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/views.html', methods=['GET', 'POST'])
def views():
    if request.method == 'POST':
        user_date = request.form['user_date']
        min_views, max_views = get_views_info(user_date)

        return render_template('views.html', user_date=user_date, min_views=min_views, max_views=max_views)

    return render_template('views.html')


def calculate_accuracy(actual, predicted):
    """
    Calculate accuracy between actual and predicted values.
    """
    return 1 - (abs(actual - predicted) / actual)

def calculate_precision(actual, predicted):
    """
    Calculate precision between actual and predicted values.
    """
    true_positives = ((actual > 0) & (predicted > 0)).sum()
    false_positives = ((actual == 0) & (predicted > 0)).sum()
    return true_positives / (true_positives + false_positives)

def calculate_recall(actual, predicted):
    """
    Calculate recall between actual and predicted values.
    """
    true_positives = ((actual > 0) & (predicted > 0)).sum()
    false_negatives = ((actual > 0) & (predicted == 0)).sum()
    return true_positives / (true_positives + false_negatives)

def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.
    """
    return 2 * (precision * recall) / (precision + recall)

def get_plot():
    train_df = pd.read_csv("train_2.csv")
    
    train_df = pd.melt(train_df[list(train_df.columns[-100:]) + ['Page']], id_vars='Page', var_name='date', value_name='Visits')
    train_df['date'] = train_df['date'].astype('datetime64[ns]')
    train_df['weekday'] = train_df['date'].apply(lambda x: x.weekday())

    df = train_df.groupby(['date']).agg({'Visits':'sum'}).rename(columns={'Visits':'visit'})

    # Load the trained SARIMAX model from the pickle file
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    model = ARIMA(train_df, order=(7,1,7))
    model_fit = model.fit()

    # Generate forecasts for the specified start date
    prediction = model_fit.forecast(len(test_df))
    prediction_value = prediction
    prediction_index = list(test_df.index)

    # Calculate Mean Squared Error
    mse = np.mean((test_df['visit'].values - prediction_value)**2)
    rmse=np.sqrt(np.sqrt(np.sqrt(mse)))
    mae = mean_absolute_error(test_df['visit'].values, prediction_value)
    
    # Calculate accuracy and precision
    accuracy = np.mean(calculate_accuracy(test_df['visit'].values, prediction_value))
    precision = np.mean(calculate_precision(test_df['visit'].values, prediction_value))
    recall = np.mean(calculate_recall(test_df['visit'].values, prediction_value))


    fig, axes = plt.subplots(figsize=(15, 8))

    axes.plot(df,label='page visit')
    axes.plot(model_fit.fittedvalues[2:], label='Predict')
    axes.plot(pd.DataFrame(model_fit.forecast(steps=len(test_df))), label='Predict', color= 'red', linestyle='--')
    axes.legend(loc='upper left')
    axes.grid()

    plot_path = 'static/forecast_plot.png'
    plt.savefig(plot_path)
    plt.close()

    return mae, mse, plot_path, accuracy, rmse

@app.route('/forecast_form.html', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        mae, mse, plot_path, accuracy,  rmse = get_plot()

        return render_template('result.html', mse=mse, rmse=rmse, mae=mae, accuracy=accuracy, encoded_plot=plot_path)
    return render_template('forecast_form.html')

@app.route('/single_forecast.html', methods=['GET', 'POST'])
def single_forecast():
    data = pd.read_csv("train_2_future.csv")
    if request.method == 'POST':
        selected_page = request.form['page']
        selected_year = int(request.form['year'])
        plot_url = plot_visits_by_month(data, selected_page, selected_year)

        return render_template('single_forecast.html', plot_url=plot_url, selected_page=selected_page, selected_year=selected_year)
    # Provide the list of pages for the dropdown menu
    pages_list = data['Page'].tolist()

    return render_template('single_forecast.html', pages_list=pages_list)

@app.route('/future_traffic')
def future_traffic():
    # Load data
    train_df = pd.read_csv("train_2_future.csv")
    X_train = train_df.drop(['Page'], axis=1)

    y = X_train.values[0]
    df1 = pd.DataFrame({ 'ds': X_train.T.index.values, 'y': y})
    # Convert dates to datetime objects with specified format
    #df1['ds'] = pd.to_datetime(df1['ds'], format='%m/%d/%Y')


    m = Prophet()
    m.fit(df1)

    # Make future dataframe
    future = m.make_future_dataframe(periods=180)
    # Predict
    forecast = m.predict(future)
    
    # Calculate accuracy and RMSE
    actual_values = X_train.values[0][-len(forecast):]  # Extract actual values for the forecast period
    predicted_values = forecast['yhat'].values[-len(forecast):]  # Extract predicted values for the forecast period

    # Ensure actual and predicted values have the same length
    if len(actual_values) != len(predicted_values):
        min_len = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[-min_len:]
        predicted_values = predicted_values[-min_len:]
        
    accuracy = np.mean(np.abs(actual_values - predicted_values) / actual_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

    # Plot and save plot
    fig = m.plot(forecast)
    fig.savefig('static/future_traffic_plot.png')  # Save plot as an image
    # Render the template with the plot image
    return render_template('future_traffic.html', plot_image='future_traffic_plot.png', accuracy=accuracy, rmse=rmse)


if __name__ == '__main__':
    app.run(debug=True)