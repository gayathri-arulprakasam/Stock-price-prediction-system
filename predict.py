import warnings
from flask import Flask, request, render_template, redirect, url_for, jsonify
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
import json  # For JSON serialization
import finnhub  # Import Finnhub
import os

# Suppress FutureWarnings from yfinance
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(_name_)

# Define holidays (example for 2025, update as necessary)
holidays = [
    datetime.date(2025, 1, 1),    # New Year's Day
    datetime.date(2025, 12, 25),  # Christmas Day
    # Add more holidays here
]

# Initialize Finnhub client
FINNHUB_API_KEY = "cuu0s5hr01qv6ijkvl60cuu0s5hr01qv6ijkvl6g"  # Fetch API key from environment variable
if not FINNHUB_API_KEY:
    raise ValueError("Finnhub API key not found. Please set the 'FINNHUB_API_KEY' environment variable.")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Helper function to check if a date is a valid weekday (not weekend or holiday)
def is_valid_weekday(date):
    return date.weekday() < 5 and date not in holidays  # Monday to Friday are valid, 0-4 represents weekdays

# Fetch stock data using yfinance
def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")  # Fetch 1 year of historical data
    if data.empty:
        raise ValueError("No data available for the selected stock ticker.")
    data.reset_index(inplace=True)
    return data

# Train a model
def train_model(data):
    # Ensure sufficient data is available
    if len(data) < 30:
        raise ValueError("Insufficient data for training. Please try another stock ticker.")

    # Convert dates to ordinal
    data['Date'] = data['Date'].map(datetime.datetime.toordinal)
    X = data[['Date']].values  # Convert to numpy array
    y = data['Close'].values   # Convert to numpy array

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data is insufficient for training the model.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Helper function to get the next valid weekday (not weekend or holiday)
def get_next_valid_day(start_date, direction=1):
    current_day = start_date
    while True:
        current_day += datetime.timedelta(days=direction)
        if is_valid_weekday(current_day):
            return current_day

# Fetch stock fundamentals using Finnhub
def fetch_fundamentals(ticker):
    try:
        profile = finnhub_client.company_profile2(symbol=ticker)
        if not profile:
            raise ValueError("No fundamental data available for the selected stock ticker.")

        fundamentals = {
            'Company Name': profile.get('name', 'N/A'),
            'Country': profile.get('country', 'N/A'),
            'Sector': profile.get('finnhubIndustry', 'N/A'),
            'Market Capitalization': profile.get('marketCapitalization', 'N/A'),
            'Industry': profile.get('finnhubIndustry', 'N/A'),
            'Number of Employees': profile.get('employees', 'N/A'),
            'IPO Date': profile.get('ipo', 'N/A'),
        }

        financials = finnhub_client.company_basic_financials(symbol=ticker, metric='all')
        if financials and 'metric' in financials and financials['metric']:
            metric = financials['metric']
            fundamentals.update({
                'Profit Margin': metric.get('profitMargin', 'N/A'),
                'Operating Margin': metric.get('operatingMargin', 'N/A'),
                'Return on Assets': metric.get('returnOnAssets', 'N/A'),
                'Return on Equity': metric.get('returnOnEquity', 'N/A'),
                'Debt to Equity': metric.get('debtToEquity', 'N/A'),
                '52-Week High': metric.get('52WeekHigh', 'N/A'),
                '52-Week Low': metric.get('52WeekLow', 'N/A'),
                'Dividend Yield': metric.get('dividendYield', 'N/A'),
                'Beta': metric.get('beta', 'N/A'),
                'Forward P/E': metric.get('forwardPE', 'N/A'),
                'PEG Ratio': metric.get('pegRatio', 'N/A'),
            })

        return fundamentals
    except Exception as e:
        print(f"Error fetching fundamentals from Finnhub: {e}")
        return {
            'Company Name': 'N/A',
            'Country': 'N/A',
            'Sector': 'N/A',
            'Market Capitalization': 'N/A',
            'Industry': 'N/A',
            'Number of Employees': 'N/A',
            'IPO Date': 'N/A',
            'Profit Margin': 'N/A',
            'Operating Margin': 'N/A',
            'Return on Assets': 'N/A',
            'Return on Equity': 'N/A',
            'Debt to Equity': 'N/A',
            '52-Week High': 'N/A',
            '52-Week Low': 'N/A',
            'Dividend Yield': 'N/A',
            'Beta': 'N/A',
            'Forward P/E': 'N/A',
            'PEG Ratio': 'N/A',
        }

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Results route
@app.route('/result', methods=['POST'])
def result():
    try:
        ticker = request.form['ticker'].upper().strip()
        data = fetch_data(ticker)
        model = train_model(data)

        current_day = datetime.datetime.now().date()

        dates = []
        predictions = []

        day_before_2 = get_next_valid_day(current_day, direction=-1)
        day_before_1 = get_next_valid_day(day_before_2, direction=-1)
        day_after_1 = get_next_valid_day(current_day, direction=1)
        day_after_2 = get_next_valid_day(day_after_1, direction=1)

        target_days = [day_before_2, day_before_1, current_day, day_after_1, day_after_2]

        for target_day in target_days:
            target_ordinal = target_day.toordinal()
            prediction = model.predict([[target_ordinal]])[0]
            formatted_date = target_day.strftime('%A, %Y-%m-%d')
            dates.append(formatted_date)
            predictions.append(round(prediction, 2))

        chart_data = {
            'dates': dates,
            'predictions': predictions
        }

        chart_data_json = json.dumps(chart_data)

        # Updated logic for buy, sell, or hold
        prediction_current = predictions[2]
        prediction_after_two_days = predictions[4]

        change_percentage = ((prediction_after_two_days - prediction_current) / prediction_current) * 100

        buy_percentage = 0
        sell_percentage = 0
        hold_percentage = 0

        if change_percentage > 0.5:
            buy_percentage = 100  # Strong buy signal
        elif change_percentage < -0.5:
            sell_percentage = 100  # Strong sell signal
        elif -0.4 <= change_percentage <= 0.4:
            hold_percentage = 100  # Hold signal

        fundamentals = fetch_fundamentals(ticker)

        return render_template('result.html',
                               ticker=ticker,
                               current_date=dates[2],
                               prediction_current=predictions[2],
                               day_after_1=dates[3],
                               prediction_after_1=predictions[3],
                               day_after_2=dates[4],
                               prediction_after_2=predictions[4],
                               chart_data=chart_data_json,
                               buy_percentage=round(buy_percentage, 2),
                               hold_percentage=round(hold_percentage, 2),
                               sell_percentage=round(sell_percentage, 2),
                               fundamentals=fundamentals)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return render_template('index.html', error='An unexpected error occurred: ' + str(e))

if _name_ == '_main_':
    app.run(debug=True)
import warnings
from flask import Flask, request, render_template, redirect, url_for, jsonify
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
import json  # For JSON serialization
import finnhub  # Import Finnhub
import os

# Suppress FutureWarnings from yfinance
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(_name_)

# Define holidays (example for 2025, update as necessary)
holidays = [
    datetime.date(2025, 1, 1),    # New Year's Day
    datetime.date(2025, 12, 25),  # Christmas Day
    # Add more holidays here
]

# Initialize Finnhub client
FINNHUB_API_KEY = "cu19io1r01qqr3sg9s1gcu19io1r01qqr3sg9s20"  # Fetch API key from environment variable
if not FINNHUB_API_KEY:
    raise ValueError("Finnhub API key not found. Please set the 'FINNHUB_API_KEY' environment variable.")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Helper function to check if a date is a valid weekday (not weekend or holiday)
def is_valid_weekday(date):
    return date.weekday() < 5 and date not in holidays  # Monday to Friday are valid, 0-4 represents weekdays

# Fetch stock data using yfinance
def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")  # Fetch 1 year of historical data
    if data.empty:
        raise ValueError("No data available for the selected stock ticker.")
    data.reset_index(inplace=True)
    return data

# Train a model
def train_model(data):
    # Ensure sufficient data is available
    if len(data) < 30:
        raise ValueError("Insufficient data for training. Please try another stock ticker.")

    # Convert dates to ordinal
    data['Date'] = data['Date'].map(datetime.datetime.toordinal)
    X = data[['Date']].values  # Convert to numpy array
    y = data['Close'].values   # Convert to numpy array

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data is insufficient for training the model.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Helper function to get the next valid weekday (not weekend or holiday)
def get_next_valid_day(start_date, direction=1):
    current_day = start_date
    while True:
        current_day += datetime.timedelta(days=direction)
        if is_valid_weekday(current_day):
            return current_day

# Fetch stock fundamentals using Finnhub
def fetch_fundamentals(ticker):
    try:
        profile = finnhub_client.company_profile2(symbol=ticker)
        if not profile:
            raise ValueError("No fundamental data available for the selected stock ticker.")

        fundamentals = {
            'Company Name': profile.get('name', 'N/A'),
            'Country': profile.get('country', 'N/A'),
            'Sector': profile.get('finnhubIndustry', 'N/A'),
            'Market Capitalization': profile.get('marketCapitalization', 'N/A'),
            'Industry': profile.get('finnhubIndustry', 'N/A'),
            'Number of Employees': profile.get('employees', 'N/A'),
            'IPO Date': profile.get('ipo', 'N/A'),
        }

        financials = finnhub_client.company_basic_financials(symbol=ticker, metric='all')
        if financials and 'metric' in financials and financials['metric']:
            metric = financials['metric']
            fundamentals.update({
                'Profit Margin': metric.get('profitMargin', 'N/A'),
                'Operating Margin': metric.get('operatingMargin', 'N/A'),
                'Return on Assets': metric.get('returnOnAssets', 'N/A'),
                'Return on Equity': metric.get('returnOnEquity', 'N/A'),
                'Debt to Equity': metric.get('debtToEquity', 'N/A'),
                '52-Week High': metric.get('52WeekHigh', 'N/A'),
                '52-Week Low': metric.get('52WeekLow', 'N/A'),
                'Dividend Yield': metric.get('dividendYield', 'N/A'),
                'Beta': metric.get('beta', 'N/A'),
                'Forward P/E': metric.get('forwardPE', 'N/A'),
                'PEG Ratio': metric.get('pegRatio', 'N/A'),
            })

        return fundamentals
    except Exception as e:
        print(f"Error fetching fundamentals from Finnhub: {e}")
        return {
            'Company Name': 'N/A',
            'Country': 'N/A',
            'Sector': 'N/A',
            'Market Capitalization': 'N/A',
            'Industry': 'N/A',
            'Number of Employees': 'N/A',
            'IPO Date': 'N/A',
            'Profit Margin': 'N/A',
            'Operating Margin': 'N/A',
            'Return on Assets': 'N/A',
            'Return on Equity': 'N/A',
            'Debt to Equity': 'N/A',
            '52-Week High': 'N/A',
            '52-Week Low': 'N/A',
            'Dividend Yield': 'N/A',
            'Beta': 'N/A',
            'Forward P/E': 'N/A',
            'PEG Ratio': 'N/A',
        }

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Results route
@app.route('/result', methods=['POST'])
def result():
    try:
        ticker = request.form['ticker'].upper().strip()
        data = fetch_data(ticker)
        model = train_model(data)

        current_day = datetime.datetime.now().date()

        dates = []
        predictions = []

        day_before_2 = get_next_valid_day(current_day, direction=-1)
        day_before_1 = get_next_valid_day(day_before_2, direction=-1)
        day_after_1 = get_next_valid_day(current_day, direction=1)
        day_after_2 = get_next_valid_day(day_after_1, direction=1)

        target_days = [day_before_2, day_before_1, current_day, day_after_1, day_after_2]

        for target_day in target_days:
            target_ordinal = target_day.toordinal()
            prediction = model.predict([[target_ordinal]])[0]
            formatted_date = target_day.strftime('%A, %Y-%m-%d')
            dates.append(formatted_date)
            predictions.append(round(prediction, 2))

        chart_data = {
            'dates': dates,
            'predictions': predictions
        }

        chart_data_json = json.dumps(chart_data)

        # Updated logic for buy, sell, or hold
        prediction_current = predictions[2]
        prediction_after_two_days = predictions[4]

        change_percentage = ((prediction_after_two_days - prediction_current) / prediction_current) * 100

        buy_percentage = 0
        sell_percentage = 0
        hold_percentage = 0

        if change_percentage > 0.5:
            buy_percentage = 100  # Strong buy signal
        elif change_percentage < -0.5:
            sell_percentage = 100  # Strong sell signal
        elif -0.4 <= change_percentage <= 0.4:
            hold_percentage = 100  # Hold signal

        fundamentals = fetch_fundamentals(ticker)

        return render_template('result.html',
                               ticker=ticker,
                               current_date=dates[2],
                               prediction_current=predictions[2],
                               day_after_1=dates[3],
                               prediction_after_1=predictions[3],
                               day_after_2=dates[4],
                               prediction_after_2=predictions[4],
                               chart_data=chart_data_json,
                               buy_percentage=round(buy_percentage, 2),
                               hold_percentage=round(hold_percentage, 2),
                               sell_percentage=round(sell_percentage, 2),
                               fundamentals=fundamentals)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return render_template('index.html', error='An unexpected error occurred: ' + str(e))

if _name_ == '_main_':
    app.run(debug=True)
