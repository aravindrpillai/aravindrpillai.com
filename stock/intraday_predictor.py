import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import date, datetime, timedelta
import pandas as pd
import glob
import json
import os

class IntradayPredictor:
    def __init__(self, ticker, period="90d", interval="1d"):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.org_data = yf.download(ticker + ".NS", period=period, interval=interval, progress=False)
        # Flatten MultiIndex columns if necessary.
        if isinstance(self.org_data.columns, pd.MultiIndex):
            self.org_data.columns = self.org_data.columns.get_level_values(0)
    
    def calculate_previous_difference(self, data):
        """
        Given a DataFrame 'data' (a slice of historical data),
        compute technical features and train a Linear Regression model
        to predict the next day's closing price.
        """
        # Create target by shifting 'Close' by -1.
        data['Next_Close'] = data['Close'].shift(-1)
        data.dropna(inplace=True)
        
        # Use OHLC as features.
        features = data[['Open', 'High', 'Low', 'Close']]
        target = data['Next_Close']
    
        # Use first 80% of the data for training.
        split_index = int(len(data) * 0.8)
        X_train = features.iloc[:split_index]
        y_train = target.iloc[:split_index]
    
        # Build and train the model.
        model = LinearRegression()
        model.fit(X_train, y_train)
    
        # Use the last row (today's data) to predict tomorrow's close.
        # Use .iloc[[-1]] to preserve feature names.
        todays_candle = features.iloc[[-1]]
        predicted_closing_price = round(model.predict(todays_candle)[0], 2)
        return predicted_closing_price
    
    def run_predictions(self, iterations=10, window=60):
        """
        Run predictions over the last 'iterations' iterations using a sliding window
        of size 'window'. Accumulates the positive and negative differences
        between predicted and actual closing prices, and computes a prediction range.
        """
        positive_variance_sum = 0
        positive_variance_count = 0
        negative_variance_sum = 0
        negative_variance_count = 0
        
        for i in range(iterations):
            actual_closing_price = round(self.org_data['Close'].iloc[-1-i], 2)
            timestamp = self.org_data.index[-1-i]
            if(i == 0):
                self.last_closing_price = actual_closing_price
                self.last_closing_date = timestamp
            
            # Select a sliding window of historical data.
            data_slice = self.org_data.iloc[(-window - i):(-1 - i)].copy()
    
            predicted_closing_price = self.calculate_previous_difference(data_slice)
            amt_diff = round(predicted_closing_price - actual_closing_price, 2)
            #print(f"{timestamp} || {self.ticker:<10} || {actual_closing_price:<9} || {predicted_closing_price:<9} || {amt_diff}")
    
            if amt_diff > 0:
                positive_variance_sum += amt_diff
                positive_variance_count += 1
            else:
                negative_variance_sum += (-amt_diff)  # take absolute value
                negative_variance_count += 1
    
        # Compute averages; handle division by zero if needed.
        negative_variance_average = (negative_variance_sum / negative_variance_count
                                     if negative_variance_count else 0)
        positive_variance_average = (positive_variance_sum / positive_variance_count
                                     if positive_variance_count else 0)
    
        # print("\nNegative Variance Average:", negative_variance_average)
        # print("Positive Variance Average:", positive_variance_average)
    
        # Predict tomorrow's closing price using the entire dataset.
        tomms_prediction = self.calculate_previous_difference(self.org_data)
        # Compute prediction range.
        pred_min = tomms_prediction - positive_variance_average
        pred_max = tomms_prediction + negative_variance_average
    
        #print("\nPredicted range for tomorrow's closing value: {} -- {}".format(pred_min, pred_max))
        return pred_min, pred_max

    @staticmethod
    def process_ticker(ticker, exchange):
        predictor = IntradayPredictor(ticker)
        pred_min, pred_max = predictor.run_predictions(iterations=10, window=30)
        
        down_variance = predictor.last_closing_price - pred_min
        up_variance = pred_max - predictor.last_closing_price
        
        up_power = up_variance / down_variance
        down_power = down_variance / up_variance

        if(predictor.last_closing_price < pred_min):
            prediction = "MUST Buy"
            description = "Lowest predicition is above last closing value"
        elif(predictor.last_closing_price > pred_max):
            prediction = "NEVER Buy"
            description = "Highest predicition is below last closing value"
        if(up_power > 3):
            prediction = "Strong Buy"
            description = "Earning power is 3 times more that loss power"
        elif(up_power > 2):
            prediction = "Buy"
            description = "Earning power is 2 times more that loss power"
        elif(up_power > 1):
            prediction = "Idle"
            description = "Profit Loss ratio almost same"
        if(down_power > 3):
            prediction = "Potential Loss"
            description = "Loss power is 3 times more that earning power"
        elif(down_power > 2):
            prediction = "Loss"
            description = "Loss power is 3 times more that earning power"
        elif(down_power > 1):
            prediction = "Idle"
            description = "Profit Loss ratio almost same"
        
        timestamp_str = str(predictor.last_closing_date)
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        last_closing_date = timestamp.date()

        #print(f"{ticker:<10} {exchange:<5} {last_closing_date} : {predictor.last_closing_price:<9} {str(round(pred_min, 2)):<9} {str(round(pred_max, 2)):<9} {prediction:<15} {description}")

        return {
            'ticker': ticker,
            'exchange' : exchange,
            "last_closing_date" : last_closing_date,
            "last_closing_price" : round(predictor.last_closing_price,2),
            'prediction_min': round(pred_min, 2),
            'prediction_max': round(pred_max, 2),
            'down_variance' : round(down_variance,2),
            "up_variance" : round(up_variance,2),
            "prediction" :prediction,
            "description" : description
        }
    
    @staticmethod
    def get_next_weekday_from_today():
        today = datetime.today()
        weekday = today.weekday()  # Get the weekday (0=Monday, 6=Sunday)
        if weekday in [0, 1, 2, 3]:  # Monday to Thursday → Next day
            next_day = today + timedelta(days=1)
        else:  # Friday, Saturday, Sunday → Next Monday
            days_to_monday = 7 - weekday
            next_day = today + timedelta(days=days_to_monday)
        return next_day.strftime("%Y-%m-%d")

    @staticmethod
    def start():
        directory = os.path.join("static", "stock", "intraday", "reports")
        os.makedirs(directory, exist_ok=True)
        tomorrow_date = IntradayPredictor.get_next_weekday_from_today()
        temp_filename = os.path.join(directory, f"{tomorrow_date}.processing")
        if os.path.exists(temp_filename):
            #Process is already running.
            return
        else: 
            with open(temp_filename, "w", encoding="utf-8") as f:
                f.write("In Progress")
        

        exchange = "NSE"
        results = []
        nse_file_path = os.path.join(os.path.join("static", "stock"), "nse.txt")
        print("Starting process....")
        with open(nse_file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    ticker = line.strip()
                    results.append(IntradayPredictor.process_ticker(ticker, exchange))
                except:
                    results.append({
                        "ticker": ticker,
                        "exchange": exchange,
                        "last_closing_date": None,
                        "last_closing_price": 0,
                        "prediction_min": 0,
                        "prediction_max": 0,
                        "down_variance": 0,
                        "up_variance": 0,
                        "prediction": "Idle",
                        "description": "Error - No information"
                    })
                    pass
        
        filename = os.path.join(directory, f"{tomorrow_date}.json")
        json_string = json.dumps(results, indent=4, default=str)
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_string)
        
        #Below code will delete old reocrds (prior to 10 days)
        cutoff_date = datetime.today() - timedelta(days=10)
        pattern = os.path.join(directory, "*.json")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            try:
                date_str = os.path.splitext(filename)[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date <= cutoff_date:
                    os.remove(file_path)
            except ValueError:
                print(f"Skipping file (invalid date format): {filename}")
        
        print("Process Completed....")