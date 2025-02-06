import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta

class StockEngine:
    def __init__(self, ticker, days_back=365):
        self.ticker = ticker + ".NS"  # .NS for NSE (National Stock Exchange)
        self.end_date = datetime.today().strftime('%Y-%m-%d')  # Today's date
        self.start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')  # Dynamic start date
        self.data = self._fetch_data()
        self.features = self._calculate_technical_indicators()
        self.model, self.accuracy = self._train_model()

    '''
    # Fetch historical stock data from Yahoo Finance.
    '''
    def _fetch_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data found for {self.ticker} between {self.start_date} and {self.end_date}.")
        return data

    """
    # Calculate technical indicators like RSI, MACD, Bollinger Bands, EMA, and SMA.
    """
    def _calculate_technical_indicators(self):
        data = self.data.copy()

        # Ensure 'Close' is a 1D Pandas Series
        close_prices = data['Close'].squeeze()  # Convert to 1D Series if necessary

        # RSI (Relative Strength Index)
        rsi = RSIIndicator(close_prices, window=14)
        data['rsi'] = rsi.rsi()

        # MACD (Moving Average Convergence Divergence)
        macd = MACD(close_prices, window_slow=26, window_fast=12, window_sign=9)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = BollingerBands(close_prices, window=20, window_dev=2)
        data['bollinger_hband'] = bollinger.bollinger_hband()
        data['bollinger_lband'] = bollinger.bollinger_lband()

        # EMA (Exponential Moving Average)
        data['ema_20'] = EMAIndicator(close_prices, window=20).ema_indicator()

        # SMA (Simple Moving Average)
        data['sma_50'] = SMAIndicator(close_prices, window=50).sma_indicator()
        data['sma_200'] = SMAIndicator(close_prices, window=200).sma_indicator()

        # Drop NaN values created by indicators
        data = data.dropna()
        return data

    """
    # Prepare features and target variable for the model.
    """
    def _prepare_features(self):
        features = self.features.copy()
        features['Target'] = np.where(features['Close'].shift(-1) > features['Close'], 1, 0)
        features = features.dropna()
        X = features[['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'ema_20', 'sma_50', 'sma_200']]
        y = features['Target']
        return X, y

    """
    # Train a RandomForestClassifier model
    """
    def _train_model(self):
        X, y = self._prepare_features()
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Not enough data to train the model. Please check the date range and stock ticker.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy

    """
    # Get benchmark and analysis for each indicator.
    """
    def _get_benchmark_and_analysis(self, indicator, current_value, close_price=None):
        benchmark = None
        analysis = ""

        if indicator == "rsi":
            benchmark = 50  # Neutral level
            if current_value > 70:
                analysis = "Overbought (Sell signal)"
            elif current_value < 30:
                analysis = "Oversold (Buy signal)"
            else:
                analysis = "Neutral"
        elif indicator == "macd":
            benchmark = 0  # MACD crossover level
            if current_value > 0:
                analysis = "Bullish momentum (Buy signal)"
            else:
                analysis = "Bearish momentum (Sell signal)"
        elif indicator == "bollinger_hband":
            benchmark = close_price  # Compare with current price
            if current_value > close_price:
                analysis = "Price near upper band (Potential resistance)"
            else:
                analysis = "Price below upper band"
        elif indicator == "bollinger_lband":
            benchmark = close_price  # Compare with current price
            if current_value < close_price:
                analysis = "Price near lower band (Potential support)"
            else:
                analysis = "Price above lower band"
        elif indicator == "ema_20":
            benchmark = close_price  # Compare with current price
            if current_value < close_price:
                analysis = "Price above EMA 20 (Bullish)"
            else:
                analysis = "Price below EMA 20 (Bearish)"
        elif indicator == "sma_50":
            benchmark = close_price  # Compare with current price
            if current_value < close_price:
                analysis = "Price above SMA 50 (Bullish)"
            else:
                analysis = "Price below SMA 50 (Bearish)"
        elif indicator == "sma_200":
            benchmark = close_price  # Compare with current price
            if current_value < close_price:
                analysis = "Price above SMA 200 (Bullish)"
            else:
                analysis = "Price below SMA 200 (Bearish)"

        return benchmark, analysis
    

    """
    # Generate prediction with dynamic JSON output
    """
    def predict_today(self):
        latest_data = self.features.iloc[-1:]
        current_price = float(latest_data['Close'].values[0])

        # Prepare JSON structure
        output = {
            "ticker": self.ticker,
            "status": True,
            "prediction_accuracy": float(self.accuracy),
            "current_price": current_price,
            "decision": "",
            "stop_loss": float(latest_data['bollinger_lband'].values[0]),
            "target_price": float(latest_data['bollinger_hband'].values[0]),
            "technical_indicators": [],
            "detailed_description": ""
        }

        # Populate technical indicators
        indicators = {
            "rsi": {"current_value": float(latest_data['rsi'].values[0])},
            "macd": {"current_value": float(latest_data['macd'].values[0])},
            "macd_signal": {"current_value": float(latest_data['macd_signal'].values[0])},
            "macd_diff": {"current_value": float(latest_data['macd_diff'].values[0])},
            "bollinger_hband": {"current_value": float(latest_data['bollinger_hband'].values[0])},
            "bollinger_lband": {"current_value": float(latest_data['bollinger_lband'].values[0])},
            "ema_20": {"current_value": float(latest_data['ema_20'].values[0])},
            "sma_50": {"current_value": float(latest_data['sma_50'].values[0])},
            "sma_200": {"current_value": float(latest_data['sma_200'].values[0])},
        }

        for indicator, values in indicators.items():
            benchmark, analysis = self._get_benchmark_and_analysis(
                indicator, values["current_value"], current_price
            )
            output["technical_indicators"].append({
                "indicator": indicator,
                "current_value": values["current_value"],
                "benchmark": benchmark,
                "analysis": analysis
            })

        # Generate dynamic description
        reasons = []
        rsi = indicators['rsi']['current_value']
        macd = indicators['macd']['current_value']
        sma_200 = indicators['sma_200']['current_value']
        bollinger_hband = indicators['bollinger_hband']['current_value']
        bollinger_lband = indicators['bollinger_lband']['current_value']
        
        if  rsi < 30:
            reasons.append(f"Oversold (RSI: {rsi:.2f} < 30)")
        elif rsi > 70:
            reasons.append(f"Overbought (RSI: {rsi:.2f} > 70)")

        if macd > 0:
            reasons.append(f"Bullish MACD crossover (MACD: {macd:.2f} > 0)")
        else:
            reasons.append(f"Bearish MACD crossover (MACD: {macd:.2f} < 0)")

        if current_price > sma_200:
            reasons.append(f"Price above 200-day SMA ({current_price:.2f} > {sma_200:.2f})")
        else:
            reasons.append(f"Price below 200-day SMA ({current_price:.2f} < {sma_200:.2f})")

        if current_price > bollinger_hband:
            reasons.append("Price above upper Bollinger Band")
        elif current_price < bollinger_lband:
            reasons.append("Price below lower Bollinger Band")

        # Final decision
        prediction = self.model.predict(latest_data[['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'ema_20', 'sma_50', 'sma_200']])
        decision = "Buy" if prediction[0] == 1 else "Sell"
        output["decision"] = decision

        # Build detailed description
        output["detailed_description"] = (
            f"Model suggests to {decision}, because:\n"
            + "\n".join([f"- {reason}" for reason in reasons])
            + f"\n\nPrediction Confidence: {self.accuracy:.2%}"
            + f"\nStop-loss: {output['stop_loss']:.2f}"
            + f"\nTarget: {output['target_price']:.2f}"
        )

        return output